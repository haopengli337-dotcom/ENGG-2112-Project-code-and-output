# =========================
# models/lstm.py
# ConvLSTM model
# Compatible with 24-hour temporal EV tensor
# =========================

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3
    ):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i_gate, f_gate, o_gate, g_gate = torch.chunk(
            gates,
            chunks=4,
            dim=1
        )

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_next = f_gate * c + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        kernel_size=3
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

    def forward(self, x):
        """
        Input:
            x shape = (B, T, C, H, W)

        Output:
            h shape = (B, hidden_channels, H, W)
        """

        B, T, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        h = torch.zeros(
            B,
            self.hidden_channels,
            H,
            W,
            device=device,
            dtype=dtype
        )

        c = torch.zeros(
            B,
            self.hidden_channels,
            H,
            W,
            device=device,
            dtype=dtype
        )

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h


if __name__ == "__main__":
    B = 1
    T = 24
    C = 6
    H = 256
    W = 256

    model = ConvLSTM(
        input_channels=C,
        hidden_channels=64
    )

    x = torch.randn(B, T, C, H, W)
    y = model(x)

    print(model)
    print("Input:", x.shape)
    print("Output:", y.shape)