"""
Vector Quantization codebook with EMA updates.

Anti-collapse mechanisms:
  - Cosine VQ: match encoder output and codebook on the unit sphere, so
    commitment loss stays bounded in [0, 4*beta/D] regardless of encoder norm.
  - Dead-code restart: replace entries with EMA cluster_size below threshold
    using random samples from the current batch latent.
  - K-means init: seed codebook from a representative pool of z_e values
    before training starts (called externally from train.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class VQCodebook(nn.Module):
    """
    EMA-updated VQ codebook.

    Args:
        num_embeddings:  codebook size K
        embedding_dim:   latent dim D
        commitment_cost: weight for the encoder commitment loss (beta)
        ema_decay:       EMA decay for cluster_size and weight buffers
        ema_update:      if True, update codebook via EMA (no autograd path);
                         if False, learn it via standard gradient with
                         codebook_loss + commitment_cost * commit_loss.
        cosine:          if True, normalize z_e and codebook to unit sphere
                         before nearest-neighbor lookup and commitment loss.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_update: bool = True,
        cosine: bool = False,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_update = ema_update
        self.cosine = cosine

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if cosine:
            # Initialize on the unit sphere (random direction per code).
            w = F.normalize(torch.randn(num_embeddings, embedding_dim), dim=-1)
            self.embedding.weight.data.copy_(w)
        else:
            nn.init.uniform_(
                self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings
            )

        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())
            self.decay = ema_decay
            self.eps = 1e-5

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-12) if self.cosine else x

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, D) continuous encoder output.

        Returns:
            z_q_st:    (B, D) quantized output with straight-through gradient.
            indices:   (B,)
            loss_vq:   scalar VQ loss.
            perplexity: scalar diagnostic.
        """
        # Distance is computed in the matching space (unit sphere if cosine).
        z_e_m = self._maybe_normalize(z_e)
        cb_m = self._maybe_normalize(self.embedding.weight)

        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2.
        # In cosine mode both norms are 1 so argmin = argmax of dot product.
        dist_mat = (
            z_e_m.pow(2).sum(1, keepdim=True)
            - 2 * z_e_m @ cb_m.t()
            + cb_m.pow(2).sum(1)
        )
        indices = dist_mat.argmin(dim=1)
        z_q = cb_m[indices]                                  # unit-norm if cosine

        # EMA update only during training, after which embedding.weight is
        # overwritten in place. Note: ema_update_buffers are all_reduced inside
        # _ema_update so all DDP ranks stay in sync.
        if self.training and self.ema_update:
            self._ema_update(z_e, indices)

        if self.ema_update:
            # EMA path: codebook follows z_e on its own, so only the encoder
            # commitment term matters. In cosine mode the loss is bounded.
            loss_vq = self.commitment_cost * F.mse_loss(z_q.detach(), z_e_m)
        else:
            loss_vq = (
                F.mse_loss(z_q.detach(), z_e_m)
                + self.commitment_cost * F.mse_loss(z_q, z_e_m.detach())
            )

        # Straight-through estimator: forward returns z_q, backward routes
        # gradients through z_e_m (and into z_e via the normalize Jacobian).
        z_q_st = z_e_m + (z_q - z_e_m).detach()

        # Perplexity = exp(entropy of usage distribution). Higher = more uniform.
        encodings = F.one_hot(indices, self.K).float()
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()

        return z_q_st, indices, loss_vq, perplexity

    # ---------------------------------------------------------------------
    # EMA update
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        z_e = z_e.detach()
        if self.cosine:
            z_e = F.normalize(z_e, dim=-1, eps=1e-12)

        encodings = F.one_hot(indices, self.K).float()
        cluster_sum = encodings.sum(0)                      # (K,)
        dw = encodings.t() @ z_e                             # (K, D)

        # DDP: aggregate per-rank statistics so every rank applies the same
        # EMA step. ema_cluster_size and ema_w then stay bit-identical.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_sum, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Laplace smoothing: prevents division by zero for unused codes.
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.K * self.eps)
            * n
        )
        new_w = self.ema_w / cluster_size.unsqueeze(1)
        if self.cosine:
            new_w = F.normalize(new_w, dim=-1, eps=1e-12)
        self.embedding.weight.data.copy_(new_w)

    # ---------------------------------------------------------------------
    # Dead-code restart
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def restart_dead_codes(
        self,
        z_e: torch.Tensor,
        threshold: float = 1.0,
    ) -> int:
        """
        Replace codes whose ema_cluster_size is below `threshold` with random
        samples from the current batch latent.

        DDP correctness:
          - ema_cluster_size is identical across ranks (already all_reduced),
            so the dead mask matches everywhere.
          - z_e differs per rank, so we all_gather the candidate pool, then
            sample on rank 0 and broadcast the chosen vectors. This keeps
            embedding.weight bit-identical across ranks.

        Returns the number of codes replaced this call.
        """
        if not self.ema_update:
            return 0

        dead = self.ema_cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        ddp = dist.is_available() and dist.is_initialized()

        z_e = z_e.detach()
        if self.cosine:
            z_e = F.normalize(z_e, dim=-1, eps=1e-12)

        # Build candidate pool. Assumes equal batch size per rank
        # (drop_last=True in DataLoader).
        if ddp:
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(z_e) for _ in range(world_size)]
            dist.all_gather(gathered, z_e)
            pool = torch.cat(gathered, dim=0)
        else:
            pool = z_e

        # Decide on rank 0, broadcast the picks.
        sampled = torch.empty(n_dead, self.D, device=z_e.device, dtype=z_e.dtype)
        if (not ddp) or dist.get_rank() == 0:
            P = pool.shape[0]
            if P == 0:
                return 0
            if P >= n_dead:
                idx = torch.randperm(P, device=pool.device)[:n_dead]
            else:
                idx = torch.randint(0, P, (n_dead,), device=pool.device)
            sampled = pool[idx].to(sampled.dtype)
        if ddp:
            dist.broadcast(sampled, src=0)

        # Replace entries and reset EMA stats so the new code lives long
        # enough to attract real assignments.
        self.embedding.weight.data[dead] = sampled
        self.ema_w.data[dead] = sampled
        self.ema_cluster_size.data[dead] = float(threshold) * 2.0
        return n_dead

    # ---------------------------------------------------------------------
    # K-means warm-up
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def kmeans_init(
        self,
        pool: torch.Tensor,
        n_iter: int = 10,
        verbose: bool = False,
    ):
        """
        Seed codebook from K-means centroids of `pool`.

        `pool` must be the same tensor on every rank (caller is responsible
        for all_gather). N >> K is recommended for stability.

        Side effects:
          embedding.weight  ← centroids
          ema_w             ← centroids
          ema_cluster_size  ← 1.0 (so the very next step's EMA can start moving)
        """
        if pool.dim() != 2 or pool.shape[1] != self.D:
            raise ValueError(
                f"kmeans pool must be (N, {self.D}); got {tuple(pool.shape)}"
            )
        if self.cosine:
            pool = F.normalize(pool, dim=-1, eps=1e-12)

        N = pool.shape[0]
        device = pool.device

        # Initial centroids: random K samples (with replacement if N < K).
        if N >= self.K:
            init_idx = torch.randperm(N, device=device)[: self.K]
        else:
            init_idx = torch.randint(0, N, (self.K,), device=device)
        centroids = pool[init_idx].clone()

        for it in range(n_iter):
            d = (
                pool.pow(2).sum(1, keepdim=True)
                - 2 * pool @ centroids.t()
                + centroids.pow(2).sum(1)
            )
            assign = d.argmin(dim=1)

            # Update centroids; empty clusters keep their previous position
            # (dead-code restart will handle them later if needed).
            one_hot = F.one_hot(assign, self.K).float()
            counts = one_hot.sum(0)
            sums = one_hot.t() @ pool
            mask = counts > 0
            new_c = centroids.clone()
            new_c[mask] = sums[mask] / counts[mask].unsqueeze(1)
            if self.cosine:
                new_c = F.normalize(new_c, dim=-1, eps=1e-12)

            if verbose:
                shift = (new_c - centroids).pow(2).mean().sqrt().item()
                print(f"  [kmeans] iter {it+1}/{n_iter} shift={shift:.4e} "
                      f"used={int(mask.sum().item())}/{self.K}")
            centroids = new_c

        self.embedding.weight.data.copy_(centroids)
        if self.ema_update:
            self.ema_w.data.copy_(centroids)
            self.ema_cluster_size.data.fill_(1.0)

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def codebook_usage(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries that appear in `indices`."""
        return indices.unique().numel() / self.K
