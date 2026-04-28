"""
Masking strategies for masked beat modeling.

Provides:
  1. Beat token masking
       - mask_beat_tokens_iid:    independent per (batch, beat, lead).
       - mask_beat_tokens_span:   contiguous-beat spans, aligned across leads.
  2. Rhythm feature masking (i.i.d. or span, mirrors the beat masking strategy).
  3. Lead dropout (entire leads zeroed across all beats), per-lead Bernoulli
     with `min_leads` floor and an optional curriculum on the dropout prob.

Cross-lead alignment (whole-beat block masking): spans are sampled per
(batch,) and broadcast across all 12 leads, so the same beat positions
are masked simultaneously across leads. ECG morphology has strong
inter-lead redundancy (12 leads view the same heartbeat from different
angles → nearly identical codebook tokens), and lead-independent masking
let the model trivially copy unmasked leads. Aligned masking forces
prediction from temporal context (other beats) and the global STFT.
"""

import math
import torch
from typing import Tuple


# -----------------------------------------------------------------------------
# Beat token masking
# -----------------------------------------------------------------------------
def mask_beat_tokens_iid(
    indices: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Independent per-position masking. Returns (masked_indices, mask).
    """
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
    """
    Per batch, pick start positions and mask `span_length` consecutive beats.
    Total fraction masked is approximately `mask_ratio`; overlapping spans
    collapse via `union`, so realized ratio is slightly lower
    (the same pattern is used by wav2vec2 / data2vec).

    cross_lead_aligned=True (default): the same beat positions are masked
    across all 12 leads (whole-beat block masking). This is required to
    prevent the model from trivially recovering masked tokens by copying
    unmasked leads at the same beat position.

    cross_lead_aligned=False: legacy lead-independent masking
    (start positions sampled per (batch, lead)).

    Edge case: if N <= span_length, mask everything.
    """
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


# -----------------------------------------------------------------------------
# Rhythm feature masking
# -----------------------------------------------------------------------------
def mask_rhythm_features(
    rr_feats: torch.Tensor,
    mask_ratio: float = 0.15,
    span_length: int = 1,
    cross_lead_aligned: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rr_feats: (B, N, L, 3). Returns (masked_rr, mask). Masked entries are
    set to zero. `span_length` >= 2 mirrors `mask_beat_tokens_span`.

    RR features are lead-invariant (the same RR triplet is repeated across
    all leads of a beat), so cross-lead-aligned masking is required for
    the rhythm task to be non-trivial; otherwise an unmasked lead at the
    same beat reveals the answer.
    """
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


# -----------------------------------------------------------------------------
# Lead dropout
# -----------------------------------------------------------------------------
def lead_dropout(
    indices: torch.Tensor,
    rr_feats: torch.Tensor,
    dropout_prob: float = 0.2,
    min_leads: int = 1,
    mask_token_id: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Drop entire leads across all beats. Per-(batch, lead) independent
    Bernoulli(dropout_prob), with `min_leads` kept per sample. The expected
    number of dropped leads is `L * dropout_prob`.

    The caller is responsible for also zeroing the corresponding lead
    channels in the STFT input passed to the model — otherwise the global
    context CNN leaks information from dropped leads.

    Returns (masked_indices, masked_rr, lead_mask=(B, L) True for dropped).
    """
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


# -----------------------------------------------------------------------------
# Lead dropout curriculum
# -----------------------------------------------------------------------------
def lead_dropout_schedule(
    epoch: int,
    max_prob: float,
    schedule: str = "linear",
    warmup_epochs: int = 50,
) -> float:
    """
    Per-epoch lead-dropout probability.

      schedule="constant": always max_prob.
      schedule="linear":   0 at epoch 1 -> max_prob at epoch warmup_epochs+1.
      schedule="cosine":   0 -> max_prob via half-cosine over warmup_epochs.
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


def mask_ratio_schedule(
    epoch: int,
    max_ratio: float,
    schedule: str = "linear",
    warmup_epochs: int = 0,
    start_ratio: float = 0.15,
) -> float:
    """Per-epoch beat / rhythm mask ratio with optional warmup.

    Same shape as `lead_dropout_schedule` but goes start_ratio → max_ratio
    over `warmup_epochs`. Set `warmup_epochs=0` (or schedule="constant") to
    disable the schedule and always emit `max_ratio` (legacy behavior).

      schedule="constant": always max_ratio.
      schedule="linear":   start_ratio at epoch 1 → max_ratio at epoch warmup+1.
      schedule="cosine":   start_ratio → max_ratio via half-cosine.
    """
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


# -----------------------------------------------------------------------------
# Combined masking
# -----------------------------------------------------------------------------
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
    """
    Full masking pipeline:
      1. lead_dropout
      2. beat token masking (span | iid), aligned across leads by default
      3. rhythm masking (mirrors the beat strategy and alignment)

    Returns a dict with masked tensors and the three masks for loss compute.
    """
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


# Backwards-compat alias for callers that imported the old name directly.
mask_beat_tokens = mask_beat_tokens_iid
