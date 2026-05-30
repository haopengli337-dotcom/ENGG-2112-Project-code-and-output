# =========================
# models/gnn.py
# Lightweight Graph Neural Network
# Compatible with 256x256 graph
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        source = edge_index[0]
        target = edge_index[1]

        N = x.shape[0]

        neighbor_sum = torch.zeros(
            N,
            x.shape[1],
            device=x.device,
            dtype=x.dtype
        )

        neighbor_count = torch.zeros(
            N,
            1,
            device=x.device,
            dtype=x.dtype
        )

        neighbor_sum.index_add_(0, target, x[source])

        neighbor_count.index_add_(
            0,
            target,
            torch.ones(
                (source.shape[0], 1),
                device=x.device,
                dtype=x.dtype
            )
        )

        neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1.0)

        out = self.self_linear(x) + self.neighbor_linear(neighbor_mean)

        return F.relu(out)


class GNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=64,
        num_layers=2
    ):
        super().__init__()

        layers = []

        if num_layers == 1:
            layers.append(GraphSAGELayer(input_dim, output_dim))
        else:
            layers.append(GraphSAGELayer(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

            layers.append(GraphSAGELayer(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, node_features, edge_index):
        x = node_features

        for layer in self.layers:
            x = layer(x, edge_index)

        return x


def node_to_grid(node_embeddings, height, width):
    N, F_dim = node_embeddings.shape

    if N != height * width:
        raise ValueError(
            f"Node count mismatch: {N} != {height * width}"
        )

    grid = node_embeddings.reshape(height, width, F_dim)
    grid = grid.permute(2, 0, 1)

    return grid


if __name__ == "__main__":
    H = 256
    W = 256
    N = H * W
    F_dim = 38

    E = 2 * ((H - 1) * W + H * (W - 1))

    node_features = torch.randn(N, F_dim)
    edge_index = torch.randint(0, N, (2, E))

    model = GNNModel(
        input_dim=F_dim,
        hidden_dim=64,
        output_dim=64,
        num_layers=2
    )

    node_embeddings = model(node_features, edge_index)

    grid_features = node_to_grid(
        node_embeddings,
        height=H,
        width=W
    )

    print(model)
    print("node_features:", node_features.shape)
    print("edge_index:", edge_index.shape)
    print("node_embeddings:", node_embeddings.shape)
    print("grid_features:", grid_features.shape)