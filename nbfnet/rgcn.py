import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass, field
from typing import List

from . import layers


@dataclass
class RGCNConfig:
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256] * 6)
    message_func: str = "distmult"
    aggregate_func: str = "pna"
    short_cut: int = 1
    layer_norm: int = 1
    activation: str = "relu"
    concat_hidden: int = 0
    # num_mlp_layer: int = 1
    dependent: int = 0


class RGCN(nn.Module):
    def __init__(
        self,
        num_relation,
        edge_embed_dim,
        cfg: RGCNConfig,
    ):
        super(RGCN, self).__init__()

        self.dims = [cfg.input_dim] + list(cfg.hidden_dims)
        self.num_relation = num_relation
        # whether to use residual connections between GNN layers
        self.short_cut = cfg.short_cut
        # whether to compute final states as a function of all layer outputs or last
        self.concat_hidden = cfg.concat_hidden
        self.edge_embed_dim = edge_embed_dim

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.RGCNConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    cfg.message_func,
                    cfg.aggregate_func,
                    cfg.layer_norm,
                    cfg.activation,
                    edge_embed_dim=edge_embed_dim,
                )
            )

        feature_dim = sum(cfg.hidden_dims) if cfg.concat_hidden else cfg.hidden_dims[-1]
        feature_dim += cfg.input_dim

        self.relation_emb = nn.Embedding(num_relation, cfg.hidden_dims[-1])
        # Final linear layer if concatenating hidden layers
        if self.concat_hidden:
            self.final_linear = nn.Linear(feature_dim, cfg.hidden_dims[-1])

    def forward(self, data, batch):
        """
        data: PyG Data object with edge indices, edge types, and optional edge attributes
        batch: Tensor of shape [batch_size, num_negative + 1, 3] containing source, relation, and target nodes
        """
        x = torch.rand(
            (data.num_nodes, self.layers[0].input_dim), device=data.edge_index.device
        )
        edge_index = data.edge_index  # edge indices of shape [2, num_edges]
        edge_type = data.edge_type  # edge types of shape [num_edges]
        edge_weight = (
            data.edge_weight if hasattr(data, "edge_weight") else None
        )  # optional edge weights

        hidden_states = []  # To store each layer's output if concat_hidden is enabled

        # Pass through each RGCN layer
        for layer in self.layers:
            new_x = layer(x, edge_index, edge_type, edge_weight)

            if self.short_cut:
                new_x = new_x + x  # Adding the shortcut connection

            x = new_x  # Update x to the new layer output
            hidden_states.append(x)  # Store for concat_hidden

        # If concatenating hidden states, combine them along the last dimension
        if self.concat_hidden:
            x = torch.cat(hidden_states, dim=-1)  # Concatenate along feature dimension
            x = self.final_linear(x)  # Reduce concatenated features to output dimension

        # Extract embeddings for each node involved in the triples in `batch`
        source_nodes = batch[:, :, 0]
        relations = batch[:, :, 1]
        target_nodes = batch[:, :, 2]

        # Gather node and relation embeddings
        source_emb = x[source_nodes]  # shape [batch_size, num_negative + 1, output_dim]
        relation_emb = self.relation_emb(
            relations
        )  # use RGCN-level relation embeddings
        target_emb = x[target_nodes]  # shape [batch_size, num_negative + 1, output_dim]

        # Compute score for each (source, relation, target) triple
        score = torch.sum(
            source_emb * relation_emb * target_emb, dim=-1
        )  # shape [batch_size, num_negative + 1]

        return score

        return scores
