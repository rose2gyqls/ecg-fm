"""
models/heads/mlm_head.py  — Masked Beat Modeling head
models/heads/classifier_head.py — Downstream classification head
(두 클래스를 한 파일에 합산)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Masked Beat Modeling Head
# ──────────────────────────────────────────────────────────────────────────────

class MaskedBeatModelingHead(nn.Module):
    """
    Transformer output -> predict masked VQ token index.

    Input : (B, S, d_model)   masked token positions
    Output: (B, S, codebook_size)  logits
    """

    def __init__(self, d_model: int = 256, codebook_size: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, codebook_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class MaskedRhythmHead(nn.Module):
    """
    Masked RR interval prediction.
    Input : (B, S, d_model)
    Output: (B, S, 3)  [prev_rr, next_rr, median_rr]
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class MaskedFiducialHead(nn.Module):
    """
    Masked Q-R / R-S interval prediction (seconds).

    Input : (B, S, d_model)  — hidden states at masked beat positions
    Output: (B, S, 2)        — [qr_sec, rs_sec]

    MLM head와 같은 beat_mask 위치에서 학습된다. Phase 1 fiducial loss가
    gradient-기반 복원 품질이었던 데 비해, Phase 3에서는 임상적 간격
    (Q-R, R-S)을 직접 regression target으로 사용해 downstream 분류와의
    signal alignment를 강화한다.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


# ──────────────────────────────────────────────────────────────────────────────
# Downstream Classifier Head
# ──────────────────────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    CLS token representation -> class logits.

    pooling: "cls" (default) | "mean"
    """

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pooling: str = "cls",
    ):
        super().__init__()
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        """
        transformer_out: (B, S, d_model)  S[0] = CLS token
        """
        if self.pooling == "cls":
            rep = transformer_out[:, 0, :]          # (B, d)
        else:
            rep = transformer_out[:, 1:, :].mean(1) # (B, d)  skip CLS
        return self.classifier(rep)
