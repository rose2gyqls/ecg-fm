import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class VQCodebook(nn.Module):
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
    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-12) if self.cosine else x
    def forward(self, z_e: torch.Tensor):
        z_e_m = self._maybe_normalize(z_e)
        cb_m = self._maybe_normalize(self.embedding.weight)
        dist_mat = (
            z_e_m.pow(2).sum(1, keepdim=True)
            - 2 * z_e_m @ cb_m.t()
            + cb_m.pow(2).sum(1)
        )
        indices = dist_mat.argmin(dim=1)
        z_q = cb_m[indices]
        if self.training and self.ema_update:
            self._ema_update(z_e, indices)
        if self.ema_update:
            loss_vq = self.commitment_cost * F.mse_loss(z_q.detach(), z_e_m)
        else:
            loss_vq = (
                F.mse_loss(z_q.detach(), z_e_m)
                + self.commitment_cost * F.mse_loss(z_q, z_e_m.detach())
            )
        z_q_st = z_e_m + (z_q - z_e_m).detach()
        sims = z_e_m @ cb_m.t()
        if self.cosine:
            tau = 0.07
            soft_logits = sims / tau
        else:
            tau = (self.D ** 0.5)
            soft_logits = sims / tau
        soft_probs = F.softmax(soft_logits, dim=-1)
        avg_soft = soft_probs.mean(0)
        neg_entropy = (avg_soft * (avg_soft + 1e-10).log()).sum()
        encodings = F.one_hot(indices, self.K).float()
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()
        return z_q_st, indices, loss_vq, perplexity, neg_entropy
    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        z_e = z_e.detach()
        if self.cosine:
            z_e = F.normalize(z_e, dim=-1, eps=1e-12)
        encodings = F.one_hot(indices, self.K).float()
        cluster_sum = encodings.sum(0)
        dw = encodings.t() @ z_e
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_sum, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
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
    @torch.no_grad()
    def restart_dead_codes(
        self,
        z_e: torch.Tensor,
        threshold: float = 1.0,
    ) -> int:
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
        if ddp:
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(z_e) for _ in range(world_size)]
            dist.all_gather(gathered, z_e)
            pool = torch.cat(gathered, dim=0)
        else:
            pool = z_e
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
        self.embedding.weight.data[dead] = sampled
        self.ema_w.data[dead] = sampled
        self.ema_cluster_size.data[dead] = float(threshold) * 2.0
        return n_dead
    @torch.no_grad()
    def kmeans_init(
        self,
        pool: torch.Tensor,
        n_iter: int = 10,
        verbose: bool = False,
    ):
        if pool.dim() != 2 or pool.shape[1] != self.D:
            raise ValueError(
                f"kmeans pool must be (N, {self.D}); got {tuple(pool.shape)}"
            )
        if self.cosine:
            pool = F.normalize(pool, dim=-1, eps=1e-12)
        N = pool.shape[0]
        device = pool.device
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
    @torch.no_grad()
    def codebook_usage(self, indices: torch.Tensor) -> float:
        return indices.unique().numel() / self.K
