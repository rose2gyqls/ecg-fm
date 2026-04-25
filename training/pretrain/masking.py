"""
training/pretrain/masking.py

Pre-training용 마스킹 전략:
1. Beat token masking — i.i.d. (legacy) 또는 span (v2 default, 인접 K-beat 묶음)
2. Rhythm vector masking
3. Lead dropout — schedule(curriculum) 지원

Notes:
  - i.i.d. per-(beat, lead) 마스킹은 ECG의 강한 redundancy(인접 beat가 거의 같은
    토큰, 같은 beat 내 lead들도 매우 유사) 때문에 너무 쉽게 풀린다.
  - span masking은 인접한 N개의 beat를 한 번에 마스킹해 적어도 beat-시간축의
    redundancy 회복을 막는다. lead-축 redundancy는 stair-step / whole-beat
    block masking(=Phase C)에서 별도로 처리한다.
"""

import math
import torch
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1a. Beat token masking — i.i.d. per (b, n, l)  (legacy)
# ──────────────────────────────────────────────────────────────────────────────

def mask_beat_tokens_iid(
    indices: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        indices   : (B, N_beats, L)  VQ indices
        mask_ratio: fraction to mask
        mask_token_id: ID to replace masked tokens with

    Returns:
        masked_indices : (B, N, L)
        mask           : (B, N, L) bool, True = masked
    """
    B, N, L = indices.shape
    mask = torch.rand(B, N, L, device=indices.device) < mask_ratio
    masked_indices = indices.clone()
    masked_indices[mask] = mask_token_id
    return masked_indices, mask


# ──────────────────────────────────────────────────────────────────────────────
# 1b. Beat token masking — span (consecutive beats), per-(b, l) independent
# ──────────────────────────────────────────────────────────────────────────────

def mask_beat_tokens_span(
    indices: torch.Tensor,
    mask_ratio: float = 0.5,
    span_length: int = 3,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per (b, l), 인접한 `span_length` 개의 beat를 한 묶음으로 마스킹.
    전체 마스킹 비율 ≈ mask_ratio.

    Implementation:
      - n_spans = ceil(N * mask_ratio / span_length) 만큼의 span을 (b, l) 별로
        독립적으로 random start 위치로 배치.
      - 겹치는 span은 그대로 union 처리 → 실제 비율이 ratio보다 약간 낮을 수도
        있지만 wav2vec2/data2vec와 같은 관행.
      - 완전 vectorized — Python loop 없음.

    Edge case:
      - N <= span_length 인 경우: 한 span을 통째로 마스킹 (ratio 1.0).
    """
    B, N, L = indices.shape
    if span_length <= 1:
        return mask_beat_tokens_iid(indices, mask_ratio, mask_token_id)
    if N <= span_length:
        # 전부 마스킹
        mask = torch.ones(B, N, L, dtype=torch.bool, device=indices.device)
        masked_indices = indices.clone()
        masked_indices[mask] = mask_token_id
        return masked_indices, mask

    n_spans = max(1, math.ceil(N * mask_ratio / span_length))
    max_start = N - span_length + 1

    # (B, n_spans, L) random start positions per (b, l)
    starts = torch.randint(0, max_start, (B, n_spans, L), device=indices.device)

    # 각 span은 [start, start+span_length) 의 N-축 범위를 마스킹
    # pos:    (1, 1, N, 1)
    # starts: (B, n_spans, 1, L)
    pos = torch.arange(N, device=indices.device).view(1, 1, N, 1)
    s = starts.unsqueeze(2)
    in_span = (pos >= s) & (pos < s + span_length)        # (B, n_spans, N, L)
    mask = in_span.any(dim=1)                              # (B, N, L)

    masked_indices = indices.clone()
    masked_indices[mask] = mask_token_id
    return masked_indices, mask


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rhythm feature masking (i.i.d. or span — 따라감)
# ──────────────────────────────────────────────────────────────────────────────

def mask_rhythm_features(
    rr_feats: torch.Tensor,
    mask_ratio: float = 0.15,
    span_length: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rr_feats : (B, N_beats, L, 3)
        span_length: 1=i.i.d., >=2=span (beat 단위로 묶음)
    Returns:
        masked_rr : same shape, masked positions set to 0
        mask      : (B, N_beats, L) bool
    """
    B, N, L, _ = rr_feats.shape
    if span_length <= 1:
        mask = torch.rand(B, N, L, device=rr_feats.device) < mask_ratio
    else:
        # Reuse span helper on a dummy index tensor
        dummy = torch.zeros(B, N, L, dtype=torch.long, device=rr_feats.device)
        _, mask = mask_beat_tokens_span(dummy, mask_ratio, span_length, mask_token_id=0)
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
        masked_indices  : (B, N, L)
        masked_rr       : (B, N, L, 3)
        lead_mask       : (B, L) bool — True = dropped lead
    """
    B, N, L = indices.shape
    lead_mask = torch.zeros(B, L, dtype=torch.bool, device=indices.device)

    if dropout_prob > 0:
        for b in range(B):
            n_drop = max(0, int(torch.rand(1).item() * L * dropout_prob))
            n_drop = min(n_drop, L - min_leads)
            if n_drop > 0:
                perm = torch.randperm(L, device=indices.device)[:n_drop]
                lead_mask[b, perm] = True

    masked_indices = indices.clone()
    masked_rr = rr_feats.clone()
    lm = lead_mask.unsqueeze(1).expand(-1, N, -1)
    masked_indices[lm] = mask_token_id
    masked_rr[lm] = 0.0
    return masked_indices, masked_rr, lead_mask


# ──────────────────────────────────────────────────────────────────────────────
# B-4: Lead-dropout curriculum helper
# ──────────────────────────────────────────────────────────────────────────────

def lead_dropout_schedule(
    epoch: int,
    max_prob: float,
    schedule: str = "linear",
    warmup_epochs: int = 50,
) -> float:
    """
    Returns lead_dropout_prob for the given epoch.

    schedule:
      - "constant": always max_prob (legacy 호환)
      - "linear":   0 at epoch 1 → max_prob at epoch (warmup_epochs+1) → constant
      - "cosine":   0 → max_prob smoothly via 1 - 0.5*(1+cos(pi*t/W))
    """
    if schedule == "constant":
        return float(max_prob)

    t = max(0, epoch - 1)
    if t >= warmup_epochs:
        return float(max_prob)

    frac = t / max(warmup_epochs, 1)
    if schedule == "linear":
        return float(max_prob) * frac
    if schedule == "cosine":
        return float(max_prob) * (1 - 0.5 * (1 + math.cos(math.pi * frac)))
    raise ValueError(f"unknown lead_dropout schedule: {schedule}")


# ──────────────────────────────────────────────────────────────────────────────
# Combined masking for one training step
# ──────────────────────────────────────────────────────────────────────────────

def apply_masking(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    beat_mask_ratio: float = 0.5,
    rhythm_mask_ratio: float = 0.5,
    span_length: int = 3,                    # 1=i.i.d. (legacy), >=2=span
    lead_dropout_prob: float = 0.0,
    lead_min_leads: int = 1,
    mask_token_id: int = 512,
    mask_strategy: str = "span",             # "iid" | "span"
) -> dict:
    """
    Apply masking pipeline:
      1. lead dropout (lead 단위 전체 제거)
      2. beat token masking (span or iid)
      3. rhythm feature masking (span or iid; beat masking과 같은 strategy)
    """
    # 1. lead dropout
    idx, rr, lead_mask = lead_dropout(
        indices, rr_feats, lead_dropout_prob, lead_min_leads, mask_token_id
    )

    # 2. beat token masking
    if mask_strategy == "iid":
        idx, beat_mask = mask_beat_tokens_iid(idx, beat_mask_ratio, mask_token_id)
        rhythm_span = 1
    elif mask_strategy == "span":
        idx, beat_mask = mask_beat_tokens_span(
            idx, beat_mask_ratio, span_length, mask_token_id,
        )
        rhythm_span = span_length
    else:
        raise ValueError(f"unknown mask_strategy: {mask_strategy}")

    # 3. rhythm masking (independently sampled, same strategy)
    rr, rhythm_mask = mask_rhythm_features(rr, rhythm_mask_ratio, span_length=rhythm_span)

    return {
        "masked_indices":   idx,
        "masked_rr_feats":  rr,
        "beat_mask":        beat_mask,
        "rhythm_mask":      rhythm_mask,
        "lead_mask":        lead_mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Backwards-compat alias — older code/tests may import mask_beat_tokens directly
# ──────────────────────────────────────────────────────────────────────────────
mask_beat_tokens = mask_beat_tokens_iid
