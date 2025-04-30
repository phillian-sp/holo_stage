import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from dataclasses import dataclass, field
from typing import List

from . import edge_rgcn_conv


@dataclass
class RGCNConfig:
    input_dim: int = 256
    num_layers: int = 6
    aggregate_func: str = "mean"
    short_cut: int = 1
    layer_norm: int = 1
    activation: str = "relu"
    concat_hidden: int = 0
    num_bases: int = 0
    use_stage: int = 1
    stage_method: str = "add"  # cat or add


class RGCN(nn.Module):
    def __init__(
        self,
        num_relation,
        edge_embed_dim,
        cfg: RGCNConfig,
    ):
        # edge_embed_dim = None
        if not cfg.use_stage:
            edge_embed_dim = None
        super(RGCN, self).__init__()
        self.dims = [cfg.input_dim] * (cfg.num_layers + 1)
        self.num_relation = num_relation
        self.short_cut = cfg.short_cut
        self.concat_hidden = cfg.concat_hidden
        self.edge_embed_dim = edge_embed_dim
        self.num_bases = cfg.num_bases

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                edge_rgcn_conv.EdgeRGCNConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    cfg.aggregate_func,
                    cfg.layer_norm,
                    cfg.activation,
                    num_bases=cfg.num_bases,
                    stage_method=cfg.stage_method,
                    edge_embed_dim=edge_embed_dim,
                )
                # RGCNConv(
                #     self.dims[i],
                #     self.dims[i + 1],
                #     num_relation,
                #     num_bases=cfg.num_bases,
                #     root_weight=True,
                #     bias=True,
                # )
            )

        # TODO: DELETE
        # self.layer_norm = nn.LayerNorm(self.dims[-1]) if cfg.layer_norm else None

        feature_dim = cfg.input_dim * cfg.num_layers

        self.relation_emb = nn.Embedding(num_relation, cfg.input_dim)

        nn.init.xavier_uniform_(self.relation_emb.weight, gain=nn.init.calculate_gain(cfg.activation))

        # Final linear layer if concatenating hidden layers
        if self.concat_hidden:
            self.final_linear = nn.Linear(feature_dim, cfg.input_dim)

        # print number of parameters in self.model
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters in RGCN: {num_params}")

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        """
        data: PyG Data object with edge indices, edge types, and optional edge attributes
        batch: Tensor of shape [batch_size, num_negative + 1, 3] containing source, relation, and target nodes
        """
        # x = torch.rand(
        #     (1, data.num_nodes, self.layers[0].input_dim), device=data.edge_index.device
        # )
        # make x constant
        x = torch.ones((1, data.num_nodes, self.dims[0]), device=data.edge_index.device)
        edge_index = data.edge_index  # edge indices of shape [2, num_edges]
        edge_type = data.original_edge_type  # edge types of shape [num_edges]
        if self.edge_embed_dim is not None:
            # edge embeddings of shape [num_edges, edge_embed_dim]
            edge_embed = data.edge_embeddings
        else:
            edge_embed = None
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

        hidden_states = []  # To store each layer's output if concat_hidden is enabled

        # Pass through each RGCN layer
        for layer in self.layers:
            new_x = layer.forward(x, edge_index, edge_type, edge_weight, edge_embed)
            # new_x = layer(x, edge_index, edge_type.long())

            # TODO: DELETE
            # if self.layer_norm:
            #     new_x = self.layer_norm(new_x)
            # new_x = torch.relu(new_x)  # Apply activation function

            if self.short_cut:
                new_x = new_x + x  # Adding the shortcut connection

            x = new_x  # Update x to the new layer output
            hidden_states.append(x)  # Store for concat_hidden

        # If concatenating hidden states, combine them along the last dimension
        if self.concat_hidden:
            x = torch.cat(hidden_states, dim=-1)  # Concatenate along feature dimension
            x = self.final_linear(x)  # Reduce concatenated features to output dimension

        x = x.expand(batch.size(0), -1, -1)

        # Extract embeddings for each node involved in the triples in `batch`
        source_nodes = batch[:, :, 0].unsqueeze(-1).expand(-1, -1, x.size(-1))
        target_nodes = batch[:, :, 1].unsqueeze(-1).expand(-1, -1, x.size(-1))
        relations = batch[:, :, 2]
        # Gather node and relation embeddings
        source_emb = x.gather(1, source_nodes)
        target_emb = x.gather(1, target_nodes)
        relation_emb = self.relation_emb(relations)

        # Compute score for each (source, relation, target) triple
        # score: [batch_size, num_negative + 1]
        score = torch.sum(source_emb * relation_emb * target_emb, dim=-1)

        # print probability use softmax
        # prob = torch.softmax(score, dim=-1)
        # print(f"prob: {prob[:, 0].mean()}")
        # # print average rank of positive samples
        # rank = torch.argsort(score, dim=-1, descending=True)
        # print(f"rank: {torch.where(rank == 0)[1].mean()}")

        return score
