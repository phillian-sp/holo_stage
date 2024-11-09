import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class RGCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, num_relation, edge_embed_dim=None, activation="relu"):
        super(RGCNLayer, self).__init__(aggr='add')  # Define the aggregation method here
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_embed_dim = edge_embed_dim

        # Node and relation transformations
        self.node_transform = nn.Linear(input_dim, output_dim)
        self.relation_embed = nn.Embedding(num_relation, output_dim)

        # Edge embedding transformation if edge embeddings are used
        if edge_embed_dim is not None:
            self.edge_transform = nn.Linear(edge_embed_dim, output_dim)

        # Activation function
        if isinstance(activation, str):
            self.activation = getattr(torch.nn.functional, activation)
        else:
            self.activation = activation

    def forward(self, x, edge_index, edge_type, edge_attr=None):
        # Transform node features
        x = self.node_transform(x)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_type=edge_type, edge_attr=edge_attr)

        # Apply activation function if specified
        if self.activation:
            out = self.activation(out)

        return out

    def message(self, x_j, edge_type, edge_attr):
        # Get relation embedding for each edge
        rel_embedding = self.relation_embed(edge_type)

        # Incorporate edge attributes if they exist
        if self.edge_embed_dim is not None and edge_attr is not None:
            edge_embedding = self.edge_transform(edge_attr)
            rel_embedding += edge_embedding

        # Compute the message as a combination of node and relation embeddings
        message = x_j + rel_embedding
        return message


class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_relations, edge_embed_dim=None, activation="relu"):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(hidden_dims)

        # Input layer
        self.layers.append(RGCNLayer(input_dim, hidden_dims[0], num_relations, edge_embed_dim, activation))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(RGCNLayer(hidden_dims[i-1], hidden_dims[i], num_relations, edge_embed_dim, activation))

        # Relation transformation layer for link prediction scoring
        self.relation_transform = nn.Embedding(num_relations, hidden_dims[-1])

    def forward(self, data, batch):
        '''
        data: PyG Data object with node features, edge indices, edge types, and optional edge attributes
        batch: Tensor h
        '''
        # Extract necessary data attributes
        x = data.x  # Node features
        edge_index = data.edge_index  # Edge indices
        edge_type = data.original_edge_type  # Edge types
        edge_attr = data.edge_embeddings if hasattr(data, 'edge_attr') else None  # Optional edge attributes

        # Pass through each RGCN layer
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_attr)

        # Get the scores for each triplet in batch
        batch_size, num_negatives_plus_one, _ = batch.size()
        source_nodes = batch[:, :, 0].view(-1)  # Flatten to shape [batch_size * (num_negative + 1)]
        relations = batch[:, :, 1].view(-1)
        target_nodes = batch[:, :, 2].view(-1)

        # Retrieve node embeddings and relation embeddings for the batch
        source_embeddings = x[source_nodes]  # Shape: [batch_size * (num_negative + 1), output_dim]
        relation_embeddings = self.relation_transform(relations)  # Shape: [batch_size * (num_negative + 1), output_dim]
        target_embeddings = x[target_nodes]  # Shape: [batch_size * (num_negative + 1), output_dim]

        # Compute link prediction scores (e.g., DistMult or TransE scoring function)
        scores = (source_embeddings * relation_embeddings * target_embeddings).sum(dim=-1)  # DistMult scoring
        scores = scores.view(batch_size, num_negatives_plus_one)  # Reshape to [batch_size, num_negative + 1]

        return scores
