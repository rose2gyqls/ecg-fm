"""
utils/metrics.py

VQ-VAE codebook 진단 및 downstream 평가 지표 헬퍼.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Codebook diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def codebook_usage_rate(indices: torch.Tensor, codebook_size: int) -> float:
    """사용된 codebook entry 비율 (0~1)."""
    return indices.unique().numel() / codebook_size


def codebook_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    """코드북 분포의 perplexity. 높을수록 고른 분포."""
    counts  = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    probs   = counts / counts.sum()
    entropy = -(probs * (probs + 1e-10).log()).sum()
    return entropy.exp().item()


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction quality
# ──────────────────────────────────────────────────────────────────────────────

def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    signal_power = (original ** 2).mean()
    noise_power  = ((original - reconstructed) ** 2).mean() + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def prd_percent(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference (PRD) — 낮을수록 좋음."""
    num = np.sqrt(((original - reconstructed) ** 2).sum())
    den = np.sqrt((original ** 2).sum()) + 1e-12
    return (num / den) * 100


# ──────────────────────────────────────────────────────────────────────────────
# Classification metrics (numpy-level)
# ──────────────────────────────────────────────────────────────────────────────

def compute_clf_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    n_classes: int,
) -> dict:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = 0.0

    return {"accuracy": acc, "f1_macro": f1, "auroc": auroc}
