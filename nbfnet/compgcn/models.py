from .helper import *
from .compgcn_conv import CompGCNConv
from .compgcn_conv_basis import CompGCNConvBasis

from dataclasses import dataclass, field


@dataclass
class CompGCNConfig:
    input_dim: int = 256
    num_layers: int = 2
    # aggregate_func: str = "mean"
    # short_cut: int = 1
    # layer_norm: int = 1
    # activation: str = "relu"
    num_bases: int = 0
    use_stage: int = 1
    score_func: str = "distmult"
    dropout: float = 0.1
    hid_drop: float = 0.3
    gamma: float = 40.0  # Margin
    bias: int = 1
    opn: str = "corr"
    edge_method: str = "add"
    # ConvE specific
    num_filt: int = 200
    hid_drop2: float = 0.3
    feat_drop: float = 0.3
    k_w: int = 16
    k_h: int = 16
    ker_sz: int = 7


class CompGCN(torch.nn.Module):
    def __init__(
        self,
        num_relation,
        edge_embed_dim,
        cfg: CompGCNConfig,
    ):
        # edge_embed_dim = None
        if not cfg.use_stage:
            edge_embed_dim = None
        super(CompGCN, self).__init__()
        self.edge_embed_dim = edge_embed_dim

        if cfg.score_func == "transe":
            self.model = CompGCN_TransE(num_relation // 2, edge_embed_dim, cfg)
        elif cfg.score_func == "distmult":
            self.model = CompGCN_DistMult(num_relation // 2, edge_embed_dim, cfg)
        elif cfg.score_func == "conve":
            self.model = CompGCN_ConvE(num_relation // 2, edge_embed_dim, cfg)
        else:
            raise NotImplementedError

    def forward(self, data, batch):
        self.num_nodes = data.num_nodes
        # set num_nodes for the model
        self.model.num_nodes = self.num_nodes
        edge_index = data.edge_index  # edge indices of shape [2, num_edges]
        edge_type = data.original_edge_type  # edge types of shape [num_edges]
        if self.edge_embed_dim is not None:
            # edge embeddings of shape [num_edges, edge_embed_dim]
            edge_embed = data.edge_embeddings
        else:
            edge_embed = None

        source_nodes = batch[:, :, 0]
        target_nodes = batch[:, :, 1]
        relations = batch[:, :, 2]

        return self.model(edge_index, edge_type, source_nodes, relations, target_nodes, edge_embed)


class BaseModel(torch.nn.Module):
    def __init__(self, cfg: CompGCNConfig):
        super(BaseModel, self).__init__()

        self.cfg = cfg
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class CompGCNBase(BaseModel):

    def __init__(self, num_rel, edge_embed_dim, cfg: CompGCNConfig):
        super(CompGCNBase, self).__init__(cfg)

        # self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        # self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        # self.device = self.edge_index.device
        # if self.cfg.edge_method == "add":
        #     self.edge_mlp = torch.nn.Linear(edge_embed_dim, self.cfg.input_dim)

        if self.cfg.num_bases > 0:
            self.init_rel = get_param((self.cfg.num_bases, self.cfg.input_dim))
        else:
            if self.cfg.score_func == "transe":
                self.init_rel = get_param((num_rel, self.cfg.input_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.cfg.input_dim))

        self.layers = torch.nn.ModuleList()
        if self.cfg.num_bases > 0:
            self.layers.append(
                CompGCNConvBasis(
                    self.cfg.input_dim, self.cfg.input_dim, num_rel, self.cfg.num_bases, act=self.act, params=self.cfg
                )
            )
        else:
            self.layers.append(
                CompGCNConv(self.cfg.input_dim, self.cfg.input_dim, num_rel, act=self.act, params=self.cfg)
            )

        for i in range(1, self.cfg.num_layers):
            self.layers.append(
                CompGCNConv(self.cfg.input_dim, self.cfg.input_dim, num_rel, act=self.act, params=self.cfg)
            )

        # self.register_parameter("bias", Parameter(torch.zeros(self.p.num_ent)))

    def forward_base(self, edge_index, edge_type, sub, rel, obj, drop, edge_embed=None):

        # r: (6, input_dim)
        r = self.init_rel if self.cfg.score_func != "transe" else torch.cat([self.init_rel, -self.init_rel], dim=0)
        init_embed = torch.ones((self.num_nodes, self.cfg.input_dim), device=sub.device)
        x = init_embed
        for layer in self.layers:
            x, r = layer(x, edge_index, edge_type, rel_embed=r, edge_embed=edge_embed)
            x = drop(x)

        # x: (num_nodes, input_dim)
        batch_size, num_neg_plus_1 = sub.size()
        sub_emb = torch.index_select(x, 0, sub.view(-1)).view(batch_size, num_neg_plus_1, -1)
        rel_emb = torch.index_select(r, 0, rel.view(-1)).view(batch_size, num_neg_plus_1, -1)
        obj_emb = torch.index_select(x, 0, obj.view(-1)).view(batch_size, num_neg_plus_1, -1)

        return sub_emb, rel_emb, obj_emb


class CompGCN_TransE(CompGCNBase):

    def __init__(self, num_rel, edge_embed_dim, cfg: CompGCNConfig):
        super(self.__class__, self).__init__(num_rel, edge_embed_dim, cfg)
        self.drop = torch.nn.Dropout(self.cfg.hid_drop)

    def forward(self, edge_index, edge_type, sub, rel, obj, edge_embed=None):

        sub_emb, rel_emb, obj_emb = self.forward_base(edge_index, edge_type, sub, rel, obj, self.drop, edge_embed)
        pred_emb = sub_emb + rel_emb

        x = self.cfg.gamma - torch.norm(pred_emb.unsqueeze(1) - obj_emb, p=1, dim=2)
        # score = torch.sigmoid(x)

        return x


class CompGCN_DistMult(CompGCNBase):

    def __init__(self, num_rel, edge_embed_dim, cfg: CompGCNConfig):
        super(self.__class__, self).__init__(num_rel, edge_embed_dim, cfg)
        self.drop = torch.nn.Dropout(self.cfg.hid_drop)

    def forward(self, edge_index, edge_type, sub, rel, obj, edge_embed=None):
        sub_emb, rel_emb, obj_emb = self.forward_base(edge_index, edge_type, sub, rel, obj, self.drop, edge_embed)
        x = sub_emb * rel_emb * obj_emb
        x = torch.sum(x, dim=2)

        return x


class CompGCN_ConvE(CompGCNBase):

    def __init__(self, num_rel, edge_embed_dim, cfg: CompGCNConfig):
        super(self.__class__, self).__init__(num_rel, edge_embed_dim, cfg)

        self.edge_embed_dim = edge_embed_dim
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.cfg.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.cfg.input_dim)

        self.hidden_drop = torch.nn.Dropout(self.cfg.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.cfg.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(
            1,
            out_channels=self.cfg.num_filt,
            kernel_size=(self.cfg.ker_sz, self.cfg.ker_sz),
            stride=1,
            padding=0,
            bias=self.cfg.bias,
        )

        flat_sz_h = int(2 * self.cfg.k_w) - self.cfg.ker_sz + 1
        flat_sz_w = self.cfg.k_h - self.cfg.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.cfg.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.cfg.input_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.cfg.input_dim)
        rel_embed = rel_embed.view(-1, 1, self.cfg.input_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.cfg.k_w, self.cfg.k_h))
        return stack_inp

    def forward(self, edge_index, edge_type, sub, rel, obj, edge_embed=None):

        sub_emb, rel_emb, obj_emb = self.forward_base(
            edge_index, edge_type, sub, rel, obj, self.hidden_drop, self.hidden_drop2, edge_embed
        )
        # print(sub_emb.shape, rel_emb.shape, obj_emb.shape)
        batch_size, num_neg_plus_1, _ = sub_emb.shape
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x).reshape(batch_size, num_neg_plus_1, -1)
        x = x * obj_emb
        x = torch.sum(x, dim=2)
        # x = torch.bmm(x, obj_emb.transpose(2, 1))
        # x += self.bias.expand_as(x)

        # score = torch.sigmoid(x)
        # print(x.shape)
        return x
