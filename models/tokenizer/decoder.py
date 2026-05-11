import torch
import torch.nn as nn
from typing import Sequence

class BeatDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        channels: Sequence[int] = (256, 128, 64, 32),
        kernel_sizes: Sequence[int] = (3, 3, 5, 7),
        beat_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.beat_length = beat_length
        self.init_len = beat_length // (2 ** len(channels))
        self.project  = nn.Linear(latent_dim, channels[0] * self.init_len)
        self.init_channels = channels[0]
        layers = []
        in_ch = channels[0]
        for out_ch, k in zip(channels[1:], kernel_sizes[1:]):
            layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=2,
                                   padding=k // 2, output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers += [
            nn.ConvTranspose1d(in_ch, 1, kernel_size=kernel_sizes[0], stride=2,
                               padding=kernel_sizes[0] // 2, output_padding=1),
        ]
        self.deconv_stack = nn.Sequential(*layers)
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        h = self.project(z_q)
        h = h.view(B, self.init_channels, self.init_len)
        out = self.deconv_stack(h)
        if out.shape[-1] != self.beat_length:
            out = out[..., :self.beat_length]
        return out
