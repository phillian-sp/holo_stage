from .helper import *
from .message_passing import MessagePassing


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_pp = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.bias:
            self.register_parameter("bias", Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed, edge_embed):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_pp_edge = edge_type[edge_type == edge_type.max()].size(0)
        num_edges = (edge_index.size(1) - num_pp_edge) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index, self.pp_index = (
            edge_index[:, :num_edges],
            edge_index[:, num_edges : 2 * num_edges],
            edge_index[:, 2 * num_edges :],
        )
        self.in_type, self.out_type, self.pp_type = (
            edge_type[:num_edges],
            edge_type[num_edges : 2 * num_edges],
            edge_type[2 * num_edges :],
        )

        if edge_embed is not None:
            self.in_embed, self.out_embed, self.pp_embed = (
                edge_embed[:num_edges],
                edge_embed[num_edges : 2 * num_edges],
                edge_embed[2 * num_edges :],
            )
        else:
            self.in_embed = self.out_embed = self.pp_embed = None

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm2(self.in_index, num_ent)
        self.out_norm = self.compute_norm2(self.out_index, num_ent)
        self.pp_norm = self.compute_norm2(self.pp_index, num_ent)

        in_res = self.propagate(
            "add",
            self.in_index,
            x=x,
            edge_type=self.in_type,
            rel_embed=rel_embed,
            edge_norm=self.in_norm,
            mode="in",
            edge_embed=self.in_embed,
        )
        loop_res = self.propagate(
            "add",
            self.loop_index,
            x=x,
            edge_type=self.loop_type,
            rel_embed=rel_embed,
            edge_norm=None,
            mode="loop",
            edge_embed=None,
        )
        out_res = self.propagate(
            "add",
            self.out_index,
            x=x,
            edge_type=self.out_type,
            rel_embed=rel_embed,
            edge_norm=self.out_norm,
            mode="out",
            edge_embed=self.out_embed,
        )
        pp_res = self.propagate(
            "add",
            self.pp_index,
            x=x,
            edge_type=self.pp_type,
            rel_embed=rel_embed,
            edge_norm=self.pp_norm,
            mode="pp",
            edge_embed=self.pp_embed,
        )
        out = (
            self.drop(in_res) * (1 / 4)
            + self.drop(out_res) * (1 / 4)
            + loop_res * (1 / 4)
            + self.drop(pp_res) * (1 / 4)
        )

        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == "corr":
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == "sub":
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == "mult":
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, edge_embed):
        weight = getattr(self, "w_{}".format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        if edge_embed is not None:
            rel_emb = rel_emb + edge_embed
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float("inf")] = 1
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def compute_norm2(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        assert deg[row].min() > 0
        norm = edge_weight / deg[row].pow(0.5)

        return norm

    def __repr__(self):
        return "{}({}, {}, num_rels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels
        )