from .helper import *
from .compgcn_conv import CompGCNConv

from dataclasses import dataclass


@dataclass
class CompGCNConfig:
    input_dim: int = 256
    num_layers: int = 2
    num_bases: int = 0
    use_stage: int = 1
    score_func: str = "distmult"
    dropout: float = 0.1
    hid_drop: float = 0.3
    gamma: float = 40.0  # Margin
    bias: int = 1
    opn: str = "corr"
    edge_method: str = "method1"

    def __post_init__(self):
        assert self.edge_method in ["method1", "method2", "method3"]
        assert self.opn in ["corr", "sub", "mult"]
        assert self.score_func in ["transe", "distmult"]
        assert self.use_stage in [0, 1]


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
        self.model = CompGCN_Dismult(num_relation // 2, edge_embed_dim, cfg)

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

        if self.cfg.num_bases > 0:
            self.init_rel = get_param((self.cfg.num_bases, self.cfg.input_dim))
        else:
            if self.cfg.score_func == "transe":
                self.init_rel = get_param((num_rel, self.cfg.input_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.cfg.input_dim))

        self.layers = torch.nn.ModuleList()

        for _ in range(0, self.cfg.num_layers):
            self.layers.append(
                CompGCNConv(self.cfg.input_dim, self.cfg.input_dim, num_rel, act=self.act, params=self.cfg)
            )

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


class CompGCN_Inner(CompGCNBase):

    def __init__(self, num_rel, edge_embed_dim, cfg: CompGCNConfig):
        super(self.__class__, self).__init__(num_rel, edge_embed_dim, cfg)
        self.drop = torch.nn.Dropout(self.cfg.hid_drop)

    def forward(self, edge_index, edge_type, sub, rel, obj, edge_embed=None):
        sub_emb, rel_emb, obj_emb = self.forward_base(edge_index, edge_type, sub, rel, obj, self.drop, edge_embed)
        x = sub_emb * rel_emb * obj_emb
        x = torch.sum(x, dim=2)

        return x
