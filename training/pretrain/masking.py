import math
import torch
from typing import Tuple

def mask_beat_tokens_iid(
    indices: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, L = indices.shape
    mask = torch.rand(B, N, L, device=indices.device) < mask_ratio
    masked = indices.clone()
    masked[mask] = mask_token_id
    return masked, mask

def mask_beat_tokens_span(
    indices: torch.Tensor,
    mask_ratio: float = 0.5,
    span_length: int = 3,
    mask_token_id: int = 512,
    cross_lead_aligned: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, L = indices.shape
    if span_length <= 1:
        return mask_beat_tokens_iid(indices, mask_ratio, mask_token_id)
    if N <= span_length:
        mask = torch.ones(B, N, L, dtype=torch.bool, device=indices.device)
        masked = indices.clone()
        masked[mask] = mask_token_id
        return masked, mask
    n_spans = max(1, math.ceil(N * mask_ratio / span_length))
    max_start = N - span_length + 1
    if cross_lead_aligned:
        starts = torch.randint(0, max_start, (B, n_spans, 1), device=indices.device)
        starts = starts.expand(B, n_spans, L)
    else:
        starts = torch.randint(0, max_start, (B, n_spans, L), device=indices.device)
    pos = torch.arange(N, device=indices.device).view(1, 1, N, 1)
    s = starts.unsqueeze(2)
    in_span = (pos >= s) & (pos < s + span_length)
    mask = in_span.any(dim=1)
    masked = indices.clone()
    masked[mask] = mask_token_id
    return masked, mask

def mask_rhythm_features(
    rr_feats: torch.Tensor,
    mask_ratio: float = 0.15,
    span_length: int = 1,
    cross_lead_aligned: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, L, _ = rr_feats.shape
    if span_length <= 1:
        if cross_lead_aligned:
            row = torch.rand(B, N, 1, device=rr_feats.device) < mask_ratio
            mask = row.expand(B, N, L)
        else:
            mask = torch.rand(B, N, L, device=rr_feats.device) < mask_ratio
    else:
        dummy = torch.zeros(B, N, L, dtype=torch.long, device=rr_feats.device)
        _, mask = mask_beat_tokens_span(
            dummy, mask_ratio, span_length, mask_token_id=0,
            cross_lead_aligned=cross_lead_aligned,
        )
    masked_rr = rr_feats.clone()
    masked_rr[mask] = 0.0
    return masked_rr, mask

def lead_dropout(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    dropout_prob: float = 0.2,
    min_leads: int = 1,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, L = indices.shape
    if dropout_prob <= 0:
        lead_mask = torch.zeros(B, L, dtype=torch.bool, device=indices.device)
    else:
        drop = torch.rand(B, L, device=indices.device) < dropout_prob
        max_drop = max(0, L - min_leads)
        n_dropped = drop.sum(dim=1)
        overflow_rows = (n_dropped > max_drop).nonzero(as_tuple=True)[0]
        for b in overflow_rows.tolist():
            dropped_idx = drop[b].nonzero(as_tuple=True)[0]
            n_un = int(n_dropped[b].item()) - max_drop
            perm = torch.randperm(dropped_idx.numel(), device=indices.device)[:n_un]
            drop[b, dropped_idx[perm]] = False
        lead_mask = drop
    masked_indices = indices.clone()
    masked_rr = rr_feats.clone()
    lm = lead_mask.unsqueeze(1).expand(-1, N, -1)
    masked_indices[lm] = mask_token_id
    masked_rr[lm] = 0.0
    return masked_indices, masked_rr, lead_mask

def lead_dropout_schedule(
    epoch: int,
    max_prob: float,
    schedule: str = "linear",
    warmup_epochs: int = 50,
) -> float:
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

def mask_ratio_schedule(
    epoch: int,
    max_ratio: float,
    schedule: str = "linear",
    warmup_epochs: int = 0,
    start_ratio: float = 0.15,
) -> float:
    if schedule == "constant" or warmup_epochs <= 0:
        return float(max_ratio)
    t = max(0, epoch - 1)
    if t >= warmup_epochs:
        return float(max_ratio)
    frac = t / max(warmup_epochs, 1)
    delta = float(max_ratio) - float(start_ratio)
    if schedule == "linear":
        return float(start_ratio) + delta * frac
    if schedule == "cosine":
        return float(start_ratio) + delta * (1 - 0.5 * (1 + math.cos(math.pi * frac)))
    raise ValueError(f"unknown mask_ratio schedule: {schedule}")

def apply_masking(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    beat_mask_ratio: float = 0.5,
    rhythm_mask_ratio: float = 0.5,
    span_length: int = 3,
    lead_dropout_prob: float = 0.0,
    lead_min_leads: int = 1,
    mask_token_id: int = 512,
    mask_strategy: str = "span",
    cross_lead_aligned: bool = True,
) -> dict:
    idx, rr, lead_mask = lead_dropout(
        indices, rr_feats, lead_dropout_prob, lead_min_leads, mask_token_id
    )
    if mask_strategy == "iid":
        idx, beat_mask = mask_beat_tokens_iid(idx, beat_mask_ratio, mask_token_id)
        rhythm_span = 1
    elif mask_strategy == "span":
        idx, beat_mask = mask_beat_tokens_span(
            idx, beat_mask_ratio, span_length, mask_token_id,
            cross_lead_aligned=cross_lead_aligned,
        )
        rhythm_span = span_length
    else:
        raise ValueError(f"unknown mask_strategy: {mask_strategy}")
    rr, rhythm_mask = mask_rhythm_features(
        rr, rhythm_mask_ratio, span_length=rhythm_span,
        cross_lead_aligned=cross_lead_aligned,
    )
    return {
        "masked_indices":  idx,
        "masked_rr_feats": rr,
        "beat_mask":       beat_mask,
        "rhythm_mask":     rhythm_mask,
        "lead_mask":       lead_mask,
    }
mask_beat_tokens = mask_beat_tokens_iid
