import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class BeatEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3, 3),
        strides: Sequence[int] = (2, 2, 2, 2),
        latent_dim: int = 256,
        dropout: float = 0.1,
        l2_normalize: bool = False,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(strides)
        self.l2_normalize = l2_normalize
        self.latent_dim = latent_dim
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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(in_ch, latent_dim)
        if l2_normalize:
            self.latent_scale = nn.Parameter(torch.tensor(latent_dim ** 0.5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stack(x)
        h = self.pool(h).squeeze(-1)
        z = self.project(h)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1) * self.latent_scale
        return z
