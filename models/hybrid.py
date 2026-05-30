# =========================
# models/hybrid.py
# CNN + ConvLSTM + GNN Hybrid Model
# =========================

import torch
import torch.nn as nn

from models.cnn import AttentionCNN
from models.lstm import ConvLSTM
from models.gnn import GNNModel
from models.gnn import node_to_grid


# =========================
# Hybrid Model
# =========================

class HybridEVModel(nn.Module):
    def __init__(
        self,
        static_channels,
        time_channels,
        gnn_input_dim,
        output_channels=5,
        cnn_features=64,
        lstm_hidden=64,
        gnn_hidden=64
    ):
        super().__init__()

        # =========================
        # CNN branch
        # =========================

        self.cnn_branch = AttentionCNN(
            input_channels=static_channels,
            output_channels=cnn_features,
            base_channels=cnn_features
        )

        # =========================
        # ConvLSTM branch
        # =========================

        self.temporal_branch = ConvLSTM(
            input_channels=time_channels,
            hidden_channels=lstm_hidden
        )

        # =========================
        # GNN branch
        # =========================

        self.gnn_branch = GNNModel(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden,
            output_dim=gnn_hidden,
            num_layers=2
        )

        # =========================
        # Fusion head
        # =========================

        fusion_channels = (
            cnn_features
            + lstm_hidden
            + gnn_hidden
        )

        self.fusion_head = nn.Sequential(
            nn.Conv2d(
                fusion_channels,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(
                64,
                output_channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x_static,
        x_time,
        node_features,
        edge_index,
        height,
        width
    ):
        """
        Inputs:

        x_static:
            (B, C_static, H, W)

        x_time:
            (B, T, C_time, H, W)

        node_features:
            (N, F)

        edge_index:
            (2, E)

        Output:
            (B, output_channels, H, W)
        """

        B = x_static.shape[0]

        # =========================
        # CNN spatial features
        # =========================

        cnn_features = self.cnn_branch(
            x_static
        )

        # =========================
        # Temporal ConvLSTM features
        # =========================

        temporal_features = self.temporal_branch(
            x_time
        )

        # =========================
        # Graph features
        # =========================

        gnn_node_embeddings = self.gnn_branch(
            node_features,
            edge_index
        )

        gnn_grid_features = node_to_grid(
            gnn_node_embeddings,
            height=height,
            width=width
        )

        # Add batch dimension
        gnn_grid_features = gnn_grid_features.unsqueeze(0)

        # Repeat for batch size
        if B > 1:
            gnn_grid_features = gnn_grid_features.repeat(
                B,
                1,
                1,
                1
            )

        # =========================
        # Feature fusion
        # =========================

        fused = torch.cat(
            [
                cnn_features,
                temporal_features,
                gnn_grid_features
            ],
            dim=1
        )

        output = self.fusion_head(
            fused
        )

        return output


# =========================
# Test
# =========================

if __name__ == "__main__":
    B = 1
    T = 24
    H = 100
    W = 100

    static_channels = 18
    time_channels = 19
    gnn_input_dim = 18

    N = H * W
    E = 39600

    x_static = torch.randn(
        B,
        static_channels,
        H,
        W
    )

    x_time = torch.randn(
        B,
        T,
        time_channels,
        H,
        W
    )

    node_features = torch.randn(
        N,
        gnn_input_dim
    )

    edge_index = torch.randint(
        0,
        N,
        (2, E)
    )

    model = HybridEVModel(
        static_channels=static_channels,
        time_channels=time_channels,
        gnn_input_dim=gnn_input_dim,
        output_channels=5
    )

    y = model(
        x_static=x_static,
        x_time=x_time,
        node_features=node_features,
        edge_index=edge_index,
        height=H,
        width=W
    )

    print(model)
    print("Output:", y.shape)