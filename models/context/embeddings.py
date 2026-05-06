"""
Token-level embedding modules used by ECGFoundationModel.

Each component projects to d_model so that the per-(beat, lead) token can be
formed by element-wise addition:
    T_{i,j} = MorphologyEmbedding(z) + RhythmMLP(rr) + LeadEmb(j) + BeatPos(i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeadEmbedding(nn.Embedding):
    """Lead index (0..n_leads-1) -> (d_model,)."""

    def __init__(self, n_leads: int = 12, d_model: int = 256):
        super().__init__(n_leads, d_model)


class BeatPositionEmbedding(nn.Embedding):
    """Beat-position index within a record (0..max_beats-1) -> (d_model,)."""

    def __init__(self, max_beats: int = 20, d_model: int = 256):
        super().__init__(max_beats, d_model)


class RhythmMLP(nn.Module):
    """
    Map [prev_rr, next_rr, median_rr] (seconds) to a d_model vector.

    Optional input normalization: pass `norm_mean` and `norm_std` to make
    forward apply z-score before the MLP. Raw RR values vary on a tight
    range (0.5-1.2s) so unnormalized inputs look near-constant to the MLP.
    The stats are registered as buffers, so they ride in state_dict and
    transfer automatically to any downstream encoder that loads the ckpt.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden: int = 128,
        d_model: int = 256,
        norm_mean: list | None = None,
        norm_std: list | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        if norm_mean is not None and norm_std is not None:
            mean_t = torch.tensor(norm_mean, dtype=torch.float32)
            std_t = torch.tensor(norm_std, dtype=torch.float32)
            if mean_t.numel() != input_dim or std_t.numel() != input_dim:
                raise ValueError(
                    f"RhythmMLP norm_{{mean,std}} length must match input_dim={input_dim}"
                )
            self.register_buffer("norm_mean", mean_t, persistent=True)
            self.register_buffer("norm_std", std_t, persistent=True)
            self._normalize = True
        else:
            self._normalize = False

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            rr = (rr - self.norm_mean) / (self.norm_std + 1e-8)
        return self.net(rr)


class GlobalContextCNN(nn.Module):
    """
    2D-CNN on the multi-lead log-magnitude STFT map.

    Input : (B, n_leads, F, T')
    Output: (B, d_model)  -- global summary vector g.
    """

    def __init__(
        self,
        in_channels: int = 12,
        channels: tuple = (16, 32, 64),
        d_model: int = 256,
    ):
        super().__init__()
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(in_ch, d_model)

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        h = self.cnn(stft)
        h = self.pool(h).flatten(1)
        return self.project(h)


class STFTGlobalEncoder(nn.Module):
    """
    v5 (MoRyECG) global STFT encoder. Same Conv2d-AdaptiveAvgPool-Linear
    skeleton as `GlobalContextCNN` but configurable channel widths so the
    encoder capacity can be cut down (default v5: [8, 16, 32]) and an
    optional STFT-bin dropout right after the input.

    The smaller capacity is one of three leakage-prevention measures listed
    in §10 of the v5 spec; combined with the lead-mask and the beat-aligned
    STFT-time mask applied by train_v5, the [GLOB] token cannot directly
    short-circuit the masked morphology codes.

    Input : (B, n_leads, F, T')
    Output: (B, d_model)
    """

    def __init__(
        self,
        in_channels: int = 12,
        channels: tuple = (8, 16, 32),
        d_model: int = 256,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dropout = float(input_dropout)
        # Dropout3d on (B, C, F, T') zeroes out random feature channels —
        # the standard 2D feature dropout adapted to the (F, T') spatial map.
        # We use it only when explicitly enabled by config.
        if self.input_dropout > 0:
            self.in_drop = nn.Dropout3d(p=self.input_dropout)
        else:
            self.in_drop = nn.Identity()
        layers = []
        in_ch = int(in_channels)
        for out_ch in channels:
            out_ch = int(out_ch)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(in_ch, d_model)

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        h = self.in_drop(stft)
        h = self.cnn(h)
        h = self.pool(h).flatten(1)
        return self.project(h)


class MorphologyEmbedding(nn.Embedding):
    """
    Codebook index -> (d_model,). Includes one extra row for the [MASK] token.
    """

    def __init__(self, codebook_size: int = 512, d_model: int = 256):
        super().__init__(codebook_size + 1, d_model)
        self.mask_token_id = codebook_size

    def get_mask_token(self, shape, device):
        return torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
