"""
models/tokenizer/encoder.py

Shared 1D-CNN Encoder: beat segment -> continuous latent vector z_e
모든 12 lead가 동일한 가중치를 공유한다.
"""

import torch
import torch.nn as nn
from typing import Sequence


class BeatEncoder(nn.Module):
    """
    Input : (B, 1, W)   — single-channel beat (W=256)
    Output: (B, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3, 3),
        strides: Sequence[int] = (2, 2, 2, 2),
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(strides)

        layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)
        # adaptive pooling -> flatten -> project
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(in_ch, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, W)"""
        h = self.conv_stack(x)       # (B, C, W')
        h = self.pool(h).squeeze(-1) # (B, C)
        return self.project(h)       # (B, latent_dim)
