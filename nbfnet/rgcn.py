import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass, field
from typing import List

from . import layers


# class RGCNLayer(MessagePassing):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         num_relation,
#         edge_embed_dim=None,
#         activation="relu",
#     ):
#         super(RGCNLayer, self).__init__(
#             aggr="add"
#         )  # Define the aggregation method here
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_relation = num_relation
#         self.edge_embed_dim = edge_embed_dim

#         # Node and relation transformations
#         self.node_transform = nn.Linear(input_dim, output_dim)
#         self.relation_embed = nn.Embedding(num_relation, output_dim)

#         # Edge embedding transformation if edge embeddings are used
#         if edge_embed_dim is not None:
#             self.edge_transform = nn.Linear(edge_embed_dim, output_dim)

#         # Activation function
#         if isinstance(activation, str):
#             self.activation = getattr(torch.nn.functional, activation)
#         else:
#             self.activation = activation

#     def forward(self, x, edge_index, edge_type, edge_attr=None):
#         # Transform node features
#         x = self.node_transform(x)

#         # Propagate messages
#         out = self.propagate(edge_index, x=x, edge_type=edge_type, edge_attr=edge_attr)

#         # Apply activation function if specified
#         if self.activation:
#             out = self.activation(out)

#         return out

#     def message(self, x_j, edge_type, edge_attr):
#         # Get relation embedding for each edge
#         rel_embedding = self.relation_embed(edge_type)

#         # Incorporate edge attributes if they exist
#         if self.edge_embed_dim is not None and edge_attr is not None:
#             edge_embedding = self.edge_transform(edge_attr)
#             rel_embedding += edge_embedding

#         # Compute the message as a combination of node and relation embeddings
#         message = x_j + rel_embedding
#         return message


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
    num_mlp_layer: int = 2
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
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    cfg.message_func,
                    cfg.aggregate_func,
                    cfg.layer_norm,
                    cfg.activation,
                    cfg.dependent,
                    edge_embed_dim=edge_embed_dim,
                )
            )

        feature_dim = sum(cfg.hidden_dims) if cfg.concat_hidden else cfg.hidden_dims[-1]
        feature_dim += cfg.input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, cfg.input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(cfg.num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, data, batch):
        """
        data: PyG Data object with node features, edge indices, edge types, and optional edge attributes
        batch: Tensor h
        """

        hiddens = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # Get the scores for each triplet in batch
        batch_size, num_negatives_plus_one, _ = batch.size()
        source_nodes = batch[:, :, 0].view(
            -1
        )  # Flatten to shape [batch_size * (num_negative + 1)]
        relations = batch[:, :, 1].view(-1)
        target_nodes = batch[:, :, 2].view(-1)

        # Retrieve node embeddings and relation embeddings for the batch
        source_embeddings = x[
            source_nodes
        ]  # Shape: [batch_size * (num_negative + 1), output_dim]
        relation_embeddings = self.relation_transform(
            relations
        )  # Shape: [batch_size * (num_negative + 1), output_dim]
        target_embeddings = x[
            target_nodes
        ]  # Shape: [batch_size * (num_negative + 1), output_dim]

        # Compute link prediction scores (e.g., DistMult or TransE scoring function)
        scores = (source_embeddings * relation_embeddings * target_embeddings).sum(
            dim=-1
        )  # DistMult scoring
        scores = scores.view(
            batch_size, num_negatives_plus_one
        )  # Reshape to [batch_size, num_negative + 1]

        return scores
