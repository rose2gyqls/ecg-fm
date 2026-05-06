"""
models/context/rr_bias.py — Pairwise RR-aware additive attention bias.

Used by the v5 (MoRyECG) inter-beat rhythm attention. Produces a per-head
additive bias b_{ij}^{RR} that the attention layer adds to its scaled dot
product score:
    A_{ij} = (Q_i K_j^T) / sqrt(d_h) + b_{ij}^{RR}

The bias depends only on the rhythm features of the two beats (i, j), so it
is computed ONCE per forward pass (outside the block loop) and shared across
all blocks in the encoder. Each head learns its own slope through the final
projection.

Pairwise feature φ_ij ∈ R^6 (matches §5 of the v5 spec):
    [prev_rr_i, prev_rr_j, |prev_rr_i − prev_rr_j|,
     Δrr_i,    Δrr_j,    |τ_i − τ_j|]
where Δrr_i = next_rr_i − prev_rr_i and τ_i = cumulative time since the
first beat in the record (= cumsum(prev_rr) starting from 0).

Padded beats carry rr_feats == 0, which gives the i-th cumulative time
τ_i = τ_{last_valid} (constant) and a non-zero pairwise feature against
valid beats — but those positions are excluded by the rhythm-attention
key_padding_mask, so the bias contribution is discarded by the softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_pairwise_rr_features(rr_feats: torch.Tensor) -> torch.Tensor:
    """Build per-pair rhythm features φ_ij from per-beat RR triplets.

    Args:
        rr_feats: (B, N, L, 3)  [prev_rr, next_rr, median_rr] in seconds.
            The triplet is lead-invariant by construction (the dataset
            replicates the same triplet across all 12 leads), so we collapse
            on the lead axis with `[:, :, 0, :]`.

    Returns:
        phi: (B, N, N, 6) float — pairwise feature for every (i, j) pair.
    """
    if rr_feats.dim() != 4:
        raise ValueError(
            f"rr_feats expected (B, N, L, 3), got {tuple(rr_feats.shape)}"
        )
    rr = rr_feats[:, :, 0, :]                         # (B, N, 3)
    prev_rr = rr[..., 0]                              # (B, N)
    next_rr = rr[..., 1]
    delta_rr = next_rr - prev_rr                      # (B, N)
    tau = torch.cumsum(prev_rr, dim=-1)               # (B, N)

    # Outer pair construction.
    pi = prev_rr[:, :, None].expand(-1, -1, prev_rr.size(-1))  # RR_i  (B, N, N)
    pj = prev_rr[:, None, :].expand(-1, prev_rr.size(-1), -1)  # RR_j
    di = delta_rr[:, :, None].expand_as(pi)
    dj = delta_rr[:, None, :].expand_as(pj)
    ti = tau[:, :, None].expand_as(pi)
    tj = tau[:, None, :].expand_as(pj)

    phi = torch.stack([
        pi,
        pj,
        (pi - pj).abs(),
        di,
        dj,
        (ti - tj).abs(),
    ], dim=-1)                                        # (B, N, N, 6)
    return phi


class PairwiseRRBiasMLP(nn.Module):
    """Map pairwise rhythm features to a per-head additive attention bias.

    Single shared MLP applied to every (i, j) pair. Output is reshaped to
    (B, num_heads, N, N) ready to be added to scaled-dot-product scores.
    Initialized so the bias starts near zero — the model recovers a plain
    self-attention prior at the start of training and only specializes if
    the rhythm features actually help.
    """

    def __init__(self, num_heads: int, hidden: int = 64,
                 input_features: int = 6, init_zero: bool = True):
        super().__init__()
        self.num_heads = int(num_heads)
        self.input_features = int(input_features)
        self.net = nn.Sequential(
            nn.Linear(self.input_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_heads),
        )
        if init_zero:
            # Zero the final projection so b^{RR} = 0 at init. This makes the
            # first forward pass behave exactly like vanilla MHA.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, rr_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rr_feats: (B, N, L, 3)
        Returns:
            bias: (B, num_heads, N, N)
        """
        phi = compute_pairwise_rr_features(rr_feats)             # (B, N, N, 6)
        bias = self.net(phi)                                      # (B, N, N, H)
        return bias.permute(0, 3, 1, 2).contiguous()              # (B, H, N, N)


def pad_rr_bias_for_glob(bias: torch.Tensor) -> torch.Tensor:
    """Insert a zero row/column for the prepended [GLOB] token.

    Args:
        bias: (B, H, N, N)
    Returns:
        bias_full: (B, H, N+1, N+1) where bias_full[..., 0, :] and
                   bias_full[..., :, 0] are zero (no rhythm prior between
                   [GLOB] and beats).
    """
    B, H, N, _ = bias.shape
    out = bias.new_zeros(B, H, N + 1, N + 1)
    out[..., 1:, 1:] = bias
    return out
