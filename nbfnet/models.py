import torch
import torch_geometric
from torch_geometric.nn import GINEConv, GCNConv
from torch_geometric.nn.pool import global_add_pool
from dataclasses import dataclass, field
from typing import List

from .nbfmodel import NBFNet


class MPNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, edge_model, edge_dim):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            if edge_model == "GINEConv":
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            elif edge_model == "GCNConv":
                self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x).relu()

        return x


@dataclass
class EdgeGraphsNBFNetConfig:
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256] * 6)
    message_func: str = "distmult"
    aggregate_func: str = "pna"
    short_cut: int = 1
    layer_norm: int = 1
    activation: str = "relu"
    concat_hidden: int = 0
    num_mlp_layer: int = 2
    dependent: int = 0
    remove_one_hop: int = 0
    num_beam: int = 10
    path_topk: int = 10

    # EdgeGraphsNBFNet specific
    edge_embed_dim: int = 256
    edge_embed_num_layers: int = 1
    edge_model: str = "GINEConv"
    use_p_value: int = 1


class EdgeGraphsNBFNet(NBFNet):
    def __init__(
        self,
        num_relation,
        cfg: EdgeGraphsNBFNetConfig,
    ):
        self.edge_embed_dim = cfg.edge_embed_dim
        super().__init__(
            cfg.input_dim,
            cfg.hidden_dims,
            num_relation,
            cfg.message_func,
            cfg.aggregate_func,
            cfg.short_cut,
            cfg.layer_norm,
            cfg.activation,
            cfg.concat_hidden,
            cfg.num_mlp_layer,
            cfg.dependent,
            cfg.remove_one_hop,
            cfg.num_beam,
            cfg.path_topk,
            cfg.edge_embed_dim,
        )
        if cfg.use_p_value:
            edge_dim = 2
        else:
            edge_dim = 1
        self.edgegraph_model = MPNN(
            input_dim=1,
            hidden_dim=cfg.edge_embed_dim,
            num_layers=cfg.edge_embed_num_layers,
            edge_model=cfg.edge_model,
            edge_dim=edge_dim,
        )
        self.up_emb = torch.nn.Embedding(
            1, cfg.edge_embed_dim
        )  # same embedding for all user product edges

        self.edge_model = cfg.edge_model
        self.use_p_value = cfg.use_p_value

    def forward(self, data, batch):
        if data.edgegraph_edge_attr.dim() == 1:
            data.edgegraph_edge_attr = data.edgegraph_edge_attr.unsqueeze(-1)
        if self.edge_model == "GCNConv":
            data.edgegraph_edge_attr = data.edgegraph_edge_attr[:, 0:1]
        h = self.edgegraph_model(
            data.edgegraph_x, data.edgegraph_edge_index, data.edgegraph_edge_attr
        )
        edgegraph_reprs = global_add_pool(h, data.edgegraph2ppedge)

        num_up_edges = data.edge_index.size(-1) - edgegraph_reprs.size(0)
        upgraph_emb = self.up_emb.weight.repeat((num_up_edges, 1))

        edge_embeddings = torch.vstack([upgraph_emb, edgegraph_reprs])

        data.edge_embeddings = edge_embeddings
        data.x = None
        return super().forward(data, batch)
