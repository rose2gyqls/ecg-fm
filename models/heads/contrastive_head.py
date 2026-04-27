"""
models/heads/contrastive_head.py — SimCLR-style contrastive auxiliary head.

Used during pretrain alongside MLM/RR/fiducial losses to push the per-record
CLS representation into a more patient-discriminative space. Two augmented
views (independent maskings) of the same record become positives; all other
records in the global (cross-rank) batch are negatives.

The projection MLP is discarded after pretraining — downstream consumes the
raw transformer CLS, not the projected embedding.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ProjectionHead(nn.Module):
    """2-layer MLP projection. SimCLR-style: nonlinear, then L2-normalize.

    Input : (B, d_in)
    Output: (B, d_out) on the unit hypersphere.
    """

    def __init__(self, d_in: int, d_hidden: int = 512, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
        self.d_out = int(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def _gather_no_grad(t: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0 across DDP ranks. Remote slices have no grad;
    the local rank's slice is replaced in-place with the (grad-carrying)
    original tensor so positives between local views still backprop."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world = dist.get_world_size()
    if world == 1:
        return t
    rank = dist.get_rank()
    bufs = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(bufs, t.contiguous().detach())
    bufs[rank] = t  # restore local with grad
    return torch.cat(bufs, dim=0)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    gather_distributed: bool = True,
) -> tuple[torch.Tensor, dict]:
    """SimCLR NT-Xent on two L2-normalized view stacks.

    Args:
        z1, z2: (B_local, d) projected embeddings of the two views.
                Must already be L2-normalized.
        temperature: τ.
        gather_distributed: if True and DDP is initialized, all_gather both
            stacks so each anchor sees 2·B_global − 2 negatives. Required
            for meaningful contrastive pretraining at small per-rank batches.

    Returns:
        loss   : scalar tensor.
        stats  : dict with diagnostic scalars (positive sim, neg sim, acc).
    """
    B = z1.shape[0]
    device = z1.device

    if gather_distributed:
        z1_all = _gather_no_grad(z1)
        z2_all = _gather_no_grad(z2)
    else:
        z1_all, z2_all = z1, z2

    # Stack: rows 0..N-1 are view-1 of each record (in rank order),
    # rows N..2N-1 are view-2. Positives lie on the +N (and -N) diagonal.
    Z = torch.cat([z1_all, z2_all], dim=0)              # (2N, d)
    N = z1_all.shape[0]
    sim = (Z @ Z.t()) / temperature                     # (2N, 2N)

    # Mask self-similarity.
    mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim.masked_fill_(mask_self, float("-inf"))

    # Build positive index: i ↔ i+N (and i+N ↔ i).
    pos_idx = torch.arange(2 * N, device=device)
    pos_idx = (pos_idx + N) % (2 * N)

    # Cross-entropy where each row is a logit vector and the target column is
    # the index of the positive partner. Negatives are the remaining 2N-2.
    # Only rows belonging to THIS rank carry gradients (others contribute to
    # the negative pool via z*_all but are detached).
    if gather_distributed and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        local_rows = torch.cat([
            torch.arange(rank * B, (rank + 1) * B, device=device),
            torch.arange(N + rank * B, N + (rank + 1) * B, device=device),
        ])
    else:
        local_rows = torch.arange(2 * N, device=device)

    logits = sim[local_rows]
    targets = pos_idx[local_rows]

    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        # Diagnostic: positive cos-sim, top-1 accuracy on the contrastive task.
        pos_sim = (sim[local_rows, targets] * temperature).mean().item()
        # Mean negative similarity: undo the positive entry, then average over
        # the remaining finite logits (self entries are -inf, masked out).
        neg_logits = logits.clone()
        neg_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
        finite = torch.isfinite(neg_logits)
        neg_sim = ((neg_logits[finite]) * temperature).mean().item()
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()

    return loss, {
        "ctr_pos_sim": pos_sim,
        "ctr_neg_sim": neg_sim,
        "ctr_acc":     acc,
    }
