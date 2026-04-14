"""
models/tokenizer/decoder.py

VQ-VAE Decoder: quantized latent -> reconstructed beat waveform
"""

import torch
import torch.nn as nn
from typing import Sequence


class BeatDecoder(nn.Module):
    """
    Input : (B, latent_dim)
    Output: (B, 1, beat_length)   (e.g. 256 samples)
    """

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

        # project latent -> first conv feature map
        # 256 샘플, stride=2 x4 => 16 frames at bottleneck
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

        # final layer: -> 1 channel
        layers += [
            nn.ConvTranspose1d(in_ch, 1, kernel_size=kernel_sizes[0], stride=2,
                               padding=kernel_sizes[0] // 2, output_padding=1),
        ]
        self.deconv_stack = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, latent_dim)"""
        B = z_q.shape[0]
        h = self.project(z_q)                          # (B, C * init_len)
        h = h.view(B, self.init_channels, self.init_len)  # (B, C, L0)
        out = self.deconv_stack(h)                     # (B, 1, ~beat_length)
        # 길이 불일치 보정
        if out.shape[-1] != self.beat_length:
            out = out[..., :self.beat_length]
        return out
