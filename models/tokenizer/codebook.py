"""
models/tokenizer/codebook.py

EMA 방식의 Vector Quantization codebook.
codebook collapse 방지를 위해 usage 모니터링 포함.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """
    EMA(Exponential Moving Average) 방식 VQ codebook.

    Args:
        num_embeddings : codebook 크기 K
        embedding_dim  : latent 차원 D
        commitment_cost: β (encoder commitment loss 가중치)
        ema_decay      : EMA decay γ
        ema_update     : True면 EMA, False면 gradient codebook update
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_update: bool = True,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_update = ema_update

        # codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w",            self.embedding.weight.data.clone())
            self.decay = ema_decay
            self.eps   = 1e-5

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e : (B, D)  continuous encoder output
        Returns:
            z_q      : (B, D)  quantized (straight-through grad)
            indices  : (B,)    codebook indices
            loss_vq  : scalar  VQ loss (commitment + codebook or just commitment)
            perplexity: scalar diagnostic
        """
        # distances  (B, K)
        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)                    # (B,)
        z_q     = self.embedding(indices)               # (B, D)

        # EMA codebook update (only during training)
        if self.training and self.ema_update:
            self._ema_update(z_e, indices)
            loss_vq = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        else:
            loss_vq = (
                F.mse_loss(z_q.detach(), z_e)          # codebook loss
                + self.commitment_cost * F.mse_loss(z_q, z_e.detach())  # commitment
            )

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # perplexity (entropy of codebook usage)
        encodings = F.one_hot(indices, self.K).float()  # (B, K)
        avg_probs  = encodings.mean(0)
        perplexity = (-( avg_probs * (avg_probs + 1e-10).log()).sum()).exp()

        return z_q_st, indices, loss_vq, perplexity

    # ── EMA helpers ──────────────────────────────────────────────────────────

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        encodings = F.one_hot(indices, self.K).float()  # (B, K)

        self.ema_cluster_size = (
            self.ema_cluster_size * self.decay + (1 - self.decay) * encodings.sum(0)
        )
        dw = encodings.t() @ z_e                        # (K, D)
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.K * self.eps)
            * n
        )
        self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

    # ── diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def codebook_usage(self, indices: torch.Tensor) -> float:
        """사용된 codebook entry 비율 반환."""
        used = indices.unique().numel()
        return used / self.K
