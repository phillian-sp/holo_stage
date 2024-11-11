import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        query_dim,
        message_func="distmult",
        aggregate_func="pna",
        layer_norm=False,
        activation="relu",
        dependent=True,
        edge_embed_dim=None,
    ):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_dim = query_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.edge_embed_dim = edge_embed_dim

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

        # define a new mlp layer and apply it to the edgegraph_embed
        if edge_embed_dim is not None:
            self.edgegraph_mlp = nn.Linear(edge_embed_dim, input_dim)

    def forward(
        self,
        input: torch.Tensor,  # (batch_size, num_nodes, input_dim)
        query: torch.Tensor,  # (batch_size, input_dim)
        boundary: torch.Tensor,  # (num_nodes, batch_size, input_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
        edge_type: torch.Tensor,  # (num_edges, 1+edge_dim)
        size: tuple[int, int],
        edge_weight: torch.Tensor,  # (num_edges,)
    ):
        """
        Args:
            input: node states at the previous layer (batch_size, num_nodes, input_dim)
            query: query relation embeddings (batch_size, input_dim)
            boundary: node states at the start of the message passing (num_nodes, batch_size, input_dim)
            edge_index: edge indices (2, num_edges)
            edge_type: edge types edge_type[:, 0] is the relation index, edge_type[:, 1:] is the edge attribute
            size: size of the graph (num_nodes, num_nodes)
            edge_weight: edge weights (num_edges,)

        Returns:
            output: updated node states (batch_size, num_nodes, output_dim)
        """
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(
                batch_size, self.num_relation, self.input_dim
            )
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(
            edge_index=edge_index,
            input=input,
            relation=relation,
            boundary=boundary,
            edge_type=edge_type,
            size=size,
            edge_weight=edge_weight,
        )
        return output

    def message(
        self,
        input_j: torch.Tensor,  # (batch_size, num_edges, input_dim) second node states
        relation: torch.Tensor,  # (batch_size, num_edges, input_dim) relation embeddings
        boundary: torch.Tensor,  # (num_nodes, batch_size, input_dim) node states at the start of the message passing
        edge_type: torch.Tensor,  # (num_edges, 1+edge_dim) edge attributes
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
        # here node_dim is -2
        if self.edge_embed_dim is not None:
            edge_type_idx = edge_type[:, 0].to(torch.long)
            relation_j = relation.index_select(self.node_dim, edge_type_idx)
            edgegraph_embed = self.edgegraph_mlp(edge_type[:, 1:])
            relation_j = relation_j + edgegraph_embed
        else:
            relation_j = relation.index_select(self.node_dim, edge_type)
        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        # (batch_size, num_edges + num_nodes, input_dim)
        message = torch.cat([message, boundary], dim=self.node_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        """
        Args:
            input: messages passed from the source nodes to the target nodes (batch_size, num_edges + num_nodes, input_dim)
                   + num_nodes for self-loops with the boundary condition
            edge_weight: edge weights (num_edges,)
            index: node indices of the first node in each edge (num_edges,)
            dim_size: number of nodes

        Returns:
            output: aggregated messages (batch_size, num_nodes, output_dim)
        """
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat(
            [index, torch.arange(dim_size, device=input.device)]
        )  # (num_edges + num_nodes,)
        edge_weight = torch.cat(
            [edge_weight, torch.ones(dim_size, device=input.device)]
        )
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            # mean: (batch_size, num_nodes, input_dim)
            mean = scatter(
                input * edge_weight,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            sq_mean = scatter(
                input**2 * edge_weight,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            max = scatter(
                input * edge_weight,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="max",
            )
            min = scatter(
                input * edge_weight,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="min",
            )
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            # features: (batch_size, num_nodes, input_dim, 4)
            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max.unsqueeze(-1),
                    min.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )
            # features: (batch_size, num_nodes, input_dim * 4)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            print(degree_out)
            scale = degree_out.log()
            scale = scale / scale.mean()
            # scales: (1, num_nodes, 3)
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )
            # output: (batch_size, num_nodes, input_dim * 4 * 3)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(
                input * edge_weight,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce=self.aggregate_func,
            )

        return output

    def update(self, update, input):
        """
        Args:
            update: aggregated messages (batch_size, num_nodes, input_dim * 4 * 3)
            input: node states at the previous layer (batch_size, num_nodes, input_dim)

        Returns:
            output: updated node states (batch_size, num_nodes, output_dim)
        """
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def message_and_aggregate(
        self,
        edge_index,
        input,
        relation,
        boundary,
        edge_type,
        edge_weight,
        index,
        dim_size,
    ):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul
            )
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul
            )
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul
            )
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul
            )
            sq_sum = generalized_rspmm(
                edge_index,
                edge_type,
                edge_weight,
                relation**2,
                input**2,
                sum="add",
                mul=mul,
            )
            max = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul
            )
            min = generalized_rspmm(
                edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul
            )
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary**2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)  # (node, batch_size * input_dim)
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max.unsqueeze(-1),
                    min.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )
            features = features.flatten(-2)  # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )  # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(
                -2
            )  # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def propagate(self, edge_index, size=None, **kwargs):

        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(
                edge_index, size, **kwargs
            )

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out


class RGCNConv(MessagePassing):
    eps = 1e-6

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        aggregate_func="mean",
        layer_norm=False,
        activation="relu",
        edge_embed_dim=None,
    ):
        super(RGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.aggregate_func = aggregate_func
        self.edge_embed_dim = edge_embed_dim

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.lin_s = nn.Linear(input_dim, output_dim)

        # define a new mlp layer and apply it to the edgegraph_embed
        if edge_embed_dim is not None:
            self.edgegraph_mlp = nn.Linear(edge_embed_dim, input_dim)

        self.lin_r = nn.ModuleList()
        for _ in range(num_relation):
            self.lin_r.append(nn.Linear(input_dim, input_dim))

    def forward(
        self,
        input: torch.Tensor,  # (batch_size, num_nodes, input_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
        edge_type: torch.Tensor,  # (num_edges, 1+edge_dim)
        edge_weight: torch.Tensor,  # (num_edges,)
    ):
        """
        Args:
            input: node states at the previous layer (batch_size, num_nodes, input_dim)
            edge_index: edge indices (2, num_edges)
            edge_type: edge types edge_type[:, 0] is the relation index, edge_type[:, 1:] is the edge attribute
            size: size of the graph (num_nodes, num_nodes)
            edge_weight: edge weights (num_edges,)

        Returns:
            output: updated node states (batch_size, num_nodes, output_dim)
        """
        num_node = input.size(self.node_dim)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        if self.edge_embed_dim is not None:
            edge_type_idx = edge_type[:, 0].to(torch.long)
            edgegraph_embed = self.edgegraph_mlp(edge_type[:, 1:])
        else:
            edge_type_idx = edge_type.to(torch.long)
            edgegraph_embed = 0

        output = self.propagate(
            edge_index=edge_index,
            input=input,
            edge_type_idx=edge_type_idx,
            edgegraph_embed=edgegraph_embed,
            size=(num_node, num_node),
            edge_weight=edge_weight,
        )
        return output

    def message(
        self,
        input_j: torch.Tensor,  # (batch_size, num_edges, input_dim) second node states
        edge_type_idx: torch.Tensor,  # (num_edges,) edge types
        edgegraph_embed: torch.Tensor,  # (num_edges, input_dim) edge attributes
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
        message = torch.zeros_like(input_j, device=input_j.device)
        for rel_type in range(len(self.lin_r)):
            mask = (edge_type_idx == rel_type).unsqueeze(0).unsqueeze(-1)
            rel_mapped = self.lin_r[rel_type](input_j)
            message += rel_mapped * mask

        message += edgegraph_embed

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        """
        Args:
            input: messages passed from the source nodes to the target nodes (batch_size, num_edges + num_nodes, input_dim)
                   + num_nodes for self-loops with the boundary condition
            edge_weight: edge weights (num_edges,)
            index: node indices of the first node in each edge (num_edges,)
            dim_size: number of nodes

        Returns:
            output: aggregated messages (batch_size, num_nodes, output_dim)
        """
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        output = scatter(
            input * edge_weight,
            index,
            dim=self.node_dim,
            dim_size=dim_size,
            reduce=self.aggregate_func,
        )

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
        output = self.lin_s(input) + update
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
