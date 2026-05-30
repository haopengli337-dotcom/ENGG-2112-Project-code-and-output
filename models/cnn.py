# =========================
# models/cnn.py
# Attention CNN model
# Compatible with 38-layer 256x256 EV spatial tensor
# =========================

import torch
import torch.nn as nn


# =========================
# Channel Attention
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()

        hidden = max(channels // ratio, 1)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.net(x)
        scale = scale.view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


# =========================
# Spatial Attention
# =========================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat(
            [avg_pool, max_pool],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# =========================
# Attention CNN
# =========================

class AttentionCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels=64,
        base_channels=64
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels // 2,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels // 2),

            nn.Conv2d(
                base_channels // 2,
                base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels)
        )

        self.channel_attention = ChannelAttention(
            base_channels
        )

        self.spatial_attention = SpatialAttention()

        self.decoder = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),

            nn.Conv2d(
                base_channels,
                output_channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Input:
            x shape = (B, C, H, W)

        Output:
            feature map shape = (B, output_channels, H, W)
        """

        x = self.encoder(x)

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        x = self.decoder(x)

        return x


# =========================
# Test
# =========================

if __name__ == "__main__":
    B = 1
    C = 38
    H = 256
    W = 256

    model = AttentionCNN(
        input_channels=C,
        output_channels=64,
        base_channels=64
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    print(model)
    print("Input:", x.shape)
    print("Output:", y.shape)