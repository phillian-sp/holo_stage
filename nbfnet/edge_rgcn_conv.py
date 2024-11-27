import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot

from typing import Optional


class EdgeRGCNConv(MessagePassing):
    eps = 1e-6

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relation: int,
        aggregate_func: str,
        layer_norm: bool,
        activation: str,
        num_bases: int = 0,
        stage_method: str = "cat",
        edge_embed_dim: Optional[int] = None,
    ):
        super(EdgeRGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.aggregate_func = aggregate_func
        self.edge_embed_dim = edge_embed_dim
        self.num_bases = num_bases
        self.stage_method = stage_method

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.lin_s = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.lin_s.weight)

        if self.stage_method == "cat":
            assert edge_embed_dim is not None
            self.lin_f = nn.Linear(edge_embed_dim + input_dim, output_dim)
            nn.init.xavier_uniform_(self.lin_f.weight)
        elif self.stage_method == "add":
            self.lin_f = nn.Identity()

        # define a new mlp layer and apply it to the edgegraph_embed
        if edge_embed_dim is not None:
            self.edgegraph_mlp = nn.Linear(edge_embed_dim, output_dim)
            nn.init.xavier_uniform_(self.edgegraph_mlp.weight)
        # self.edgegraph_mlp = edgegraph_mlp

        if num_bases > 0:
            self.weight = nn.Parameter(torch.empty(num_bases, input_dim, output_dim))
            self.comp = nn.Parameter(torch.empty(num_relation, num_bases))
            glorot(self.weight)
            glorot(self.comp)
        else:
            self.lin_r = nn.ModuleList()
            for _ in range(num_relation):
                self.lin_r.append(nn.Linear(input_dim, output_dim))
                nn.init.xavier_uniform_(self.lin_r[-1].weight)

    def forward(
        self,
        input: torch.Tensor,  # (batch_size, num_nodes, input_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
        edge_type: torch.Tensor,  # (num_edges,)
        edge_weight: torch.Tensor,  # (num_edges,)
        edge_embed: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input: node states at the previous layer (batch_size, num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
            edge_type: edge types edge_type[:, 0] is the relation index, edge_type[:, 1:] is the edge attribute
            edge_weight: edge weights (num_edges,)

        Returns:
            output: updated node states (batch_size, num_nodes, output_dim)
        """
        num_node = input.size(self.node_dim)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        edge_type = edge_type.to(torch.long)
        if self.edge_embed_dim is None:
            assert edge_embed is None
            edge_embed = 0
        else:
            assert edge_embed is not None

        output = self.propagate(
            edge_index=edge_index,
            input=input,
            edge_type=edge_type,
            edge_embed=edge_embed,
            size=(num_node, num_node),
            edge_weight=edge_weight,
        )
        return output

    def message(
        self,
        input_j: torch.Tensor,  # (batch_size, num_edges, input_dim) second node states
        edge_type: torch.Tensor,  # (num_edges,) edge types
        edge_embed: torch.Tensor,  # (num_edges, input_dim) edge attributes
    ):
        """
        Args:
            input_j: node states at the previous layer (batch_size, num_edges, input_dim)
            relation: relation embeddings for each edge type (batch_size, num_relation, input_dim)
            boundary: node states at the start of the message passing (num_nodes, batch_size, input_dim)
            edge_type: edge types edge_type[:, 0] is the relation index, edge_type[:, 1:] is the edge attribute

        Returns:
            message: messages passed from the source nodes to the target nodes (batch_size, num_edges + num_nodes, input_dim)
        """
        batch_size, num_edges, _ = input_j.size()
        message = torch.zeros((batch_size, num_edges, self.output_dim), device=input_j.device)
        if self.num_bases > 0:
            weight = (self.comp @ self.weight.view(self.num_bases, -1)).view(
                self.num_relation, self.input_dim, self.output_dim
            )
        for rel_type in range(self.num_relation):
            mask = (edge_type == rel_type).unsqueeze(0).unsqueeze(-1)
            # rel_mapped: (batch_size, num_edges, self.output_dim)
            if self.num_bases > 0:
                rel_mapped = torch.matmul(
                    input_j,
                    weight[rel_type],
                )
            else:
                rel_mapped = self.lin_r[rel_type](input_j)
            message += rel_mapped * mask
        if self.edge_embed_dim is not None:
            transformed_edge_embed = self.edgegraph_mlp.forward(edge_embed)
        else:
            transformed_edge_embed = 0
        if self.stage_method == "cat":
            message = torch.cat([message, transformed_edge_embed.unsqueeze(0)], dim=-1)
        elif self.stage_method == "add":
            message += transformed_edge_embed
        else:
            raise ValueError("Unknown concatenation method `%s`" % self.stage_method)

        return message

    def aggregate(self, input, edge_weight, index, edge_type, dim_size):
        """
        Args:
            input: messages passed from the source nodes to the target nodes (batch_size, num_edges, input_dim)
            edge_weight: edge weights (num_edges,)
            index: node indices of the first node in each edge (num_edges,)
            edge_type: edge type indices for each edge (num_edges,)
            dim_size: number of nodes

        Returns:
            output: aggregated messages (batch_size, num_nodes, output_dim)
        """
        # Weighted input
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)
        weighted_input = input * edge_weight

        # Initialize the output to accumulate messages from all edge types
        output = torch.zeros((input.size(0), dim_size, input.size(2)), device=input.device)

        # Loop over each unique edge type
        for rel_type in range(self.num_relation):
            # Mask to select messages of the current edge type
            type_mask = edge_type == rel_type

            # Apply the mask and perform scatter aggregation for the selected type
            aggregated_type = scatter(
                weighted_input[type_mask.unsqueeze(0)],
                index[type_mask],
                dim=self.node_dim,
                dim_size=dim_size,
                reduce=self.aggregate_func,
            )

            # Add the aggregated result for this type to the final output
            output += aggregated_type

        return output

    def update(self, update, input):
        """
        Args:
            update: aggregated messages (batch_size, num_nodes, input_dim)
            input: node states at the previous layer (batch_size, num_nodes, input_dim)

        Returns:
            output: updated node states (batch_size, num_nodes, output_dim)
        """
        # node update as a function of old states (input) and this layer output (update)
        output = self.lin_s(input) + self.lin_f(update)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
