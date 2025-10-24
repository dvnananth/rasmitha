from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
        )
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        h = self.net(x)
        y = self.out(h)
        return y.transpose(1, 2)  # (B, T, 1)


class ResidualPINN(nn.Module):
    """PINN-like residual learner that predicts r_t from windowed features.

    Physics is enforced via losses, not internal hard constraints here.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.temporal = TemporalConvBlock(input_dim, hidden_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        # x_window: (B, T, D)
        return self.temporal(x_window).squeeze(-1)  # (B, T) residual per step


class PhysicsAwareAttention(nn.Module):
    def __init__(self, feature_dim: int, num_components: int, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_components),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, F)
        logits = self.net(context) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        return weights  # (B, num_components)
