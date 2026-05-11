from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ProjectionHead(nn.Module):
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
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world = dist.get_world_size()
    if world == 1:
        return t
    rank = dist.get_rank()
    bufs = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(bufs, t.contiguous().detach())
    bufs[rank] = t
    return torch.cat(bufs, dim=0)

def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    gather_distributed: bool = True,
) -> tuple[torch.Tensor, dict]:
    B = z1.shape[0]
    device = z1.device
    if gather_distributed:
        z1_all = _gather_no_grad(z1)
        z2_all = _gather_no_grad(z2)
    else:
        z1_all, z2_all = z1, z2
    Z = torch.cat([z1_all, z2_all], dim=0)
    N = z1_all.shape[0]
    sim = (Z @ Z.t()) / temperature
    mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim.masked_fill_(mask_self, float("-inf"))
    pos_idx = torch.arange(2 * N, device=device)
    pos_idx = (pos_idx + N) % (2 * N)
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
        pos_sim = (sim[local_rows, targets] * temperature).mean().item()
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
