import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GCNConv


def tied_topk_indices(values, k, expansion=2):
    """
    >>> tied_topk_indices(torch.tensor([4,4,4,5,5]), 2, 2).sort()[0]
    tensor([3, 4])
    >>> tied_topk_indices(torch.tensor([4,1,4,5,5,1]), 3, 2).sort()[0]
    tensor([0, 2, 3, 4])
    """
    assert len(values) >= k * expansion, f"len(values)={len(values)} < k*expansion={k*expansion}"

    values, indices = torch.topk(values, k * expansion)
    assert values[k - 1] != values[-1], "Cannot break ties within expansion.\n" "Try a larger expansion value"

    return indices[: k + ((values[k - 1] == values[k:]).sum())]


class PowerMethod(torch.nn.Module):
    def __init__(self, k, out_dim):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

    def forward(self, v0, adj_t):
        v = v0
        for _ in range(self.k):
            v = adj_t.matmul(v)
        return v


class SymmetryBreakingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels, hidden_channels, normalize=False
        )  # Note: This assumes the adj matrix is normalized
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.out_dim = hidden_channels

    def forward(self, v0, adj_t):
        x = self.conv1(v0, adj_t).relu()
        return self.conv2(x, adj_t)


class Holo(torch.nn.Module):
    def __init__(self, *, n_breakings: int, symmetric_breaking_model):
        super().__init__()
        self.n_breakings = n_breakings
        self.symmetric_breaking_model = symmetric_breaking_model

        self.ln = torch.nn.LayerNorm(symmetric_breaking_model.out_dim)

    def get_nodes_to_break(self, adj_t, n_breakings=8):
        node_degrees = adj_t.sum(-1)
        return tied_topk_indices(node_degrees, k=n_breakings)

    def forward(self, X, adj_t, group_idx=None):
        # X: (n, f - 1), adj_t: (n, n)

        # break_node_indices: (t,) where b is the number of nodes to break
        break_node_indices = self.get_nodes_to_break(adj_t, n_breakings=self.n_breakings)

        # one_hot_breakings: (t, n, 1)
        one_hot_breakings = F.one_hot(break_node_indices, X.size(0)).unsqueeze(-1)
        # holo_repr: (t, n, f)
        holo_repr = self.symmetric_breaking_model(
            torch.cat(
                (
                    X.unsqueeze(0).repeat(one_hot_breakings.size(0), 1, 1),  # (t, n, f - 1)
                    one_hot_breakings,  # (t, n, 1)
                ),
                dim=-1,
            ),
            adj_t,
        )  # (t, n, f), where n includes both movies and users
        holo_repr = self.ln(holo_repr)  # (t, n, f)

        if group_idx is not None:
            holo_repr = torch_scatter.scatter(holo_repr, group_idx, 0, reduce="mean")  # (l, n, f)
        else:
            holo_repr = holo_repr.mean(0, keepdim=True)  # (l=1, n, f)

        return holo_repr.transpose(0, 1).flatten(1, 2)  # (n, f*l)
