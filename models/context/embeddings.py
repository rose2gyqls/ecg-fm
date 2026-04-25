"""
models/context/embeddings.py

Lead embedding, positional embedding, rhythm MLP, global context CNN.
모두 d_model 차원으로 projection하여 element-wise addition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# Lead Identity Embedding
# ──────────────────────────────────────────────────────────────────────────────

class LeadEmbedding(nn.Embedding):
    """
    12 leads (0~11) -> (d_model,)
    """
    def __init__(self, n_leads: int = 12, d_model: int = 256):
        super().__init__(n_leads, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Beat Position Embedding  (learned)
# ──────────────────────────────────────────────────────────────────────────────

class BeatPositionEmbedding(nn.Embedding):
    """
    10초 내 beat 순서 (0 ~ max_beats-1) -> (d_model,)
    """
    def __init__(self, max_beats: int = 20, d_model: int = 256):
        super().__init__(max_beats, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Rhythm Vector MLP   p = MLP([prev_rr, next_rr, median_rr])
# ──────────────────────────────────────────────────────────────────────────────

class RhythmMLP(nn.Module):
    """
    Input : (B, 3)  — [prev_rr, next_rr, median_rr]  in seconds
    Output: (B, d_model)

    Normalization:
        norm_mean / norm_std (length=input_dim)을 주면 forward에서 z-score 한 뒤
        MLP에 넣는다. raw RR 값(0.5~1.2초)은 표준편차가 작아 MLP 입력으로 거의
        constant처럼 보이는 문제를 보완한다. 버퍼로 등록되어 state_dict에 저장
        → downstream(benchmark)에서도 동일한 정규화가 자동 적용됨.
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
            # buffer로 등록 → state_dict에 저장 + .to(device) 자동 따라감
            self.register_buffer("norm_mean", mean_t, persistent=True)
            self.register_buffer("norm_std", std_t, persistent=True)
            self._normalize = True
        else:
            self._normalize = False

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            # rr shape: (..., input_dim). mean/std broadcast.
            rr = (rr - self.norm_mean) / (self.norm_std + 1e-8)
        return self.net(rr)


# ──────────────────────────────────────────────────────────────────────────────
# Global Context  g = 2D-CNN(STFT_map)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalContextCNN(nn.Module):
    """
    Input : (B, 12, F, T')  — 12-lead log-magnitude STFT
    Output: (B, d_model)    — global frequency summary vector g
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
        self.cnn     = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(in_ch, d_model)

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """stft: (B, 12, F, T')"""
        h = self.cnn(stft)              # (B, C, F', T'')
        h = self.pool(h).flatten(1)     # (B, C)
        return self.project(h)          # (B, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Morphology token embedding  (codebook index -> d_model)
# ──────────────────────────────────────────────────────────────────────────────

class MorphologyEmbedding(nn.Embedding):
    """
    VQ index (0~K-1) -> (d_model,)
    """
    def __init__(self, codebook_size: int = 512, d_model: int = 256):
        super().__init__(codebook_size + 1, d_model)  # +1 for [MASK] token
        self.mask_token_id = codebook_size

    def get_mask_token(self, shape, device):
        return torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
