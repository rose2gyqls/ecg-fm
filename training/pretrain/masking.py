"""
training/pretrain/masking.py

Pre-training용 마스킹 전략:
1. Beat token masking (Masked Beat Modeling)
2. Rhythm vector masking
3. Lead dropout (lead 단위 전체 제거)
"""

import torch
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1. Beat token masking
# ──────────────────────────────────────────────────────────────────────────────

def mask_beat_tokens(
    indices: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token_id: int = 512,       # codebook_size (MASK token)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        indices   : (B, N_beats, 12)  VQ indices
        mask_ratio: fraction to mask
        mask_token_id: ID to replace masked tokens with

    Returns:
        masked_indices : (B, N_beats, 12)  with some replaced by mask_token_id
        mask           : (B, N_beats, 12) bool, True = masked
    """
    B, N, L = indices.shape
    mask = torch.rand(B, N, L, device=indices.device) < mask_ratio
    masked_indices = indices.clone()
    masked_indices[mask] = mask_token_id
    return masked_indices, mask


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rhythm feature masking
# ──────────────────────────────────────────────────────────────────────────────

def mask_rhythm_features(
    rr_feats: torch.Tensor,
    mask_ratio: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rr_feats : (B, N_beats, 12, 3)
    Returns:
        masked_rr : same shape, masked positions set to 0
        mask      : (B, N_beats, 12) bool
    """
    B, N, L, _ = rr_feats.shape
    mask = torch.rand(B, N, L, device=rr_feats.device) < mask_ratio
    masked_rr = rr_feats.clone()
    masked_rr[mask] = 0.0
    return masked_rr, mask


# ──────────────────────────────────────────────────────────────────────────────
# 3. Lead dropout
# ──────────────────────────────────────────────────────────────────────────────

def lead_dropout(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    dropout_prob: float = 0.2,
    min_leads: int = 1,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    리드 단위로 전체 비트를 마스킹.

    Returns:
        masked_indices  : (B, N, 12)
        masked_rr       : (B, N, 12, 3)
        lead_mask       : (B, 12) bool — True = dropped lead
    """
    B, N, L = indices.shape
    lead_mask = torch.zeros(B, L, dtype=torch.bool, device=indices.device)

    for b in range(B):
        # 최소 min_leads는 보존
        n_drop = max(0, int(torch.rand(1).item() * L * dropout_prob))
        n_drop = min(n_drop, L - min_leads)
        if n_drop > 0:
            perm = torch.randperm(L, device=indices.device)[:n_drop]
            lead_mask[b, perm] = True

    # apply mask
    masked_indices = indices.clone()
    masked_rr      = rr_feats.clone()

    # lead_mask: (B, 12) -> expand to (B, N, 12)
    lm = lead_mask.unsqueeze(1).expand(-1, N, -1)
    masked_indices[lm] = mask_token_id
    masked_rr[lm]      = 0.0

    return masked_indices, masked_rr, lead_mask


# ──────────────────────────────────────────────────────────────────────────────
# Combined masking for one training step
# ──────────────────────────────────────────────────────────────────────────────

def apply_masking(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    beat_mask_ratio: float = 0.15,
    rhythm_mask_ratio: float = 0.15,
    lead_dropout_prob: float = 0.2,
    lead_min_leads: int = 1,
    mask_token_id: int = 512,
) -> dict:
    """
    모든 masking을 순서대로 적용하고 결과 dict 반환.
    """
    # 1. lead dropout (먼저 — beat/rhythm mask와 독립적으로 표시)
    idx, rr, lead_mask = lead_dropout(
        indices, rr_feats, lead_dropout_prob, lead_min_leads, mask_token_id
    )

    # 2. beat token masking
    idx, beat_mask = mask_beat_tokens(idx, beat_mask_ratio, mask_token_id)

    # 3. rhythm masking
    rr, rhythm_mask = mask_rhythm_features(rr, rhythm_mask_ratio)

    return {
        "masked_indices":   idx,
        "masked_rr_feats":  rr,
        "beat_mask":        beat_mask,
        "rhythm_mask":      rhythm_mask,
        "lead_mask":        lead_mask,
    }
