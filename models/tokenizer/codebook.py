"""
models/tokenizer/codebook.py

EMA 방식의 Vector Quantization codebook.
codebook collapse 방지:
  - cosine VQ (L2-normalized): encoder/codebook을 단위 구 위에서 매칭 →
    z_e norm 표류로 인한 commitment loss 우상향을 차단
  - dead-code restart: 사용량 < threshold인 entry를 batch latent에서 swap
  - kmeans_init: 첫 epoch 전 z_e 풀로 K-means 초기화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class VQCodebook(nn.Module):
    """
    EMA(Exponential Moving Average) 방식 VQ codebook.

    Args:
        num_embeddings : codebook 크기 K
        embedding_dim  : latent 차원 D
        commitment_cost: β (encoder commitment loss 가중치)
        ema_decay      : EMA decay γ
        ema_update     : True면 EMA, False면 gradient codebook update
        cosine         : True면 z_e/codebook을 unit-norm으로 매칭 (cosine 거리).
                         encoder가 l2_normalize된 z_e를 내보낼 때만 의미가 있다.
                         단, 이 모듈도 자체적으로 normalize하므로 encoder L2가
                         꺼져 있어도 동작은 한다.
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

        # codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if cosine:
            # unit-norm 분포로 시작 (단위 구에서 뽑은 random direction)
            w = torch.randn(num_embeddings, embedding_dim)
            w = F.normalize(w, dim=-1)
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

    # ── helpers ──────────────────────────────────────────────────────────────
    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.cosine:
            return F.normalize(x, dim=-1, eps=1e-12)
        return x

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e : (B, D)  continuous encoder output
        Returns:
            z_q_st   : (B, D)  quantized w/ straight-through grad
            indices  : (B,)
            loss_vq  : scalar  VQ loss
            perplexity: scalar diagnostic
        """
        # 거리 계산용 표현 (cosine이면 단위 norm, 아니면 raw)
        z_e_m = self._maybe_normalize(z_e)
        cb_m = self._maybe_normalize(self.embedding.weight)

        # squared L2 in the matching space.
        # cosine 모드에선 ‖a‖=‖b‖=1 이므로 ‖a−b‖² = 2 − 2·cos_sim → argmin = argmax cos.
        dist_mat = (
            z_e_m.pow(2).sum(1, keepdim=True)
            - 2 * z_e_m @ cb_m.t()
            + cb_m.pow(2).sum(1)
        )
        indices = dist_mat.argmin(dim=1)                # (B,)
        # 출력/commitment 모두 매칭 공간에서 일관되게 산출
        z_q = cb_m[indices]                             # (B, D), cosine이면 unit-norm

        # EMA codebook update (training만)
        if self.training and self.ema_update:
            self._ema_update(z_e, indices)

        if self.ema_update:
            # EMA 모드에서는 codebook 자체가 EMA로 갱신되므로 commitment만.
            # cosine 모드에서는 z_e_m vs z_q 모두 unit-norm이라 loss가 [0,4]로 bounded.
            loss_vq = self.commitment_cost * F.mse_loss(z_q.detach(), z_e_m)
        else:
            loss_vq = (
                F.mse_loss(z_q.detach(), z_e_m)
                + self.commitment_cost * F.mse_loss(z_q, z_e_m.detach())
            )

        # straight-through estimator. gradient는 z_e_m으로 흐른다.
        # cosine 모드에선 z_q가 unit-norm이라 decoder 입력 magnitude가 일관.
        z_q_st = z_e_m + (z_q - z_e_m).detach()

        # perplexity (entropy of codebook usage)
        encodings = F.one_hot(indices, self.K).float()
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()

        return z_q_st, indices, loss_vq, perplexity

    # ── EMA helpers ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        # EMA buffer는 cosine과 무관하게 raw z_e의 통계로 추적한다.
        # 단, cosine일 때는 매 update 후 entry를 unit-norm으로 renormalize.
        z_e = z_e.detach()
        if self.cosine:
            z_e = F.normalize(z_e, dim=-1, eps=1e-12)

        encodings = F.one_hot(indices, self.K).float()
        cluster_sum = encodings.sum(0)              # (K,)
        dw = encodings.t() @ z_e                     # (K, D)

        # DDP: 모든 rank의 로컬 배치 통계를 합산해 동일한 EMA 업데이트를 적용
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_sum, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Laplace smoothing
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

    # ── A-2: dead-code restart ───────────────────────────────────────────────

    @torch.no_grad()
    def restart_dead_codes(
        self,
        z_e: torch.Tensor,
        threshold: float = 1.0,
    ) -> int:
        """
        ema_cluster_size < threshold 인 entry를 batch latent z_e의 random sample로 swap.

        DDP 안전성:
          - ema_cluster_size는 EMA all_reduce로 모든 rank가 동일 → dead 마스크도 동일.
          - 그러나 z_e는 rank마다 다르므로, 후보 풀을 all_gather한 뒤 rank 0에서
            (n_dead)개 샘플링해 broadcast해야 모든 rank의 codebook이 비트단위로 일치.

        반환: 이번에 restart된 코드 개수.
        """
        if not self.ema_update:
            return 0

        dead = self.ema_cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        ddp = dist.is_available() and dist.is_initialized()

        # 후보 풀 구성 — DDP면 모든 rank의 z_e를 모아 더 다양한 표본 확보
        z_e = z_e.detach()
        if self.cosine:
            z_e = F.normalize(z_e, dim=-1, eps=1e-12)

        if ddp:
            world_size = dist.get_world_size()
            # 모든 rank의 batch size가 같다고 가정 (drop_last=True). 다르면 padding 필요.
            gathered = [torch.zeros_like(z_e) for _ in range(world_size)]
            dist.all_gather(gathered, z_e)
            pool = torch.cat(gathered, dim=0)
        else:
            pool = z_e

        # rank 0에서 결정 → broadcast (모든 rank 결과 일치 보장)
        sampled = torch.empty(n_dead, self.D, device=z_e.device, dtype=z_e.dtype)
        if (not ddp) or dist.get_rank() == 0:
            P = pool.shape[0]
            if P == 0:
                return 0
            if P >= n_dead:
                idx = torch.randperm(P, device=pool.device)[:n_dead]
            else:
                # 풀이 작으면 with-replacement
                idx = torch.randint(0, P, (n_dead,), device=pool.device)
            sampled = pool[idx].to(sampled.dtype)
        if ddp:
            dist.broadcast(sampled, src=0)

        # entry 교체. EMA 통계도 같이 리셋해 dead 상태에서 빠져나오게.
        self.embedding.weight.data[dead] = sampled
        self.ema_w.data[dead] = sampled
        # 새 entry가 즉시 또 dead로 분류되지 않도록 cluster_size를 충분히 띄워둠
        self.ema_cluster_size.data[dead] = float(threshold) * 2.0

        return n_dead

    # ── A-3: k-means warm-up init ────────────────────────────────────────────

    @torch.no_grad()
    def kmeans_init(
        self,
        pool: torch.Tensor,
        n_iter: int = 10,
        verbose: bool = False,
    ):
        """
        pool: (N, D) 모든 rank에서 동일한 latent 풀 (호출자가 all_gather 후 전달).
        N >= K 권장.

        결과:
          - embedding.weight = K-means centroids
          - ema_w           = centroids * (1 회분 cluster_size 가중치)
          - ema_cluster_size = 1.0 (각 코드가 살아 있다고 표시)
          - _initialized    = 1
        """
        if pool.dim() != 2 or pool.shape[1] != self.D:
            raise ValueError(
                f"kmeans pool must be (N, {self.D}); got {tuple(pool.shape)}"
            )
        if self.cosine:
            pool = F.normalize(pool, dim=-1, eps=1e-12)

        N = pool.shape[0]
        device = pool.device

        # 초기 centroid: random K samples (with replacement if N < K)
        if N >= self.K:
            init_idx = torch.randperm(N, device=device)[: self.K]
        else:
            init_idx = torch.randint(0, N, (self.K,), device=device)
        centroids = pool[init_idx].clone()  # (K, D)

        for it in range(n_iter):
            # assign — squared L2 (cosine 모드에선 둘 다 unit-norm이라 동치)
            d = (
                pool.pow(2).sum(1, keepdim=True)
                - 2 * pool @ centroids.t()
                + centroids.pow(2).sum(1)
            )
            assign = d.argmin(dim=1)                        # (N,)

            # update — 빈 클러스터는 그대로 두고, 차후 dead-code restart가 처리
            one_hot = F.one_hot(assign, self.K).float()
            counts = one_hot.sum(0)                          # (K,)
            sums = one_hot.t() @ pool                        # (K, D)
            mask = counts > 0
            new_c = centroids.clone()
            new_c[mask] = sums[mask] / counts[mask].unsqueeze(1)
            if self.cosine:
                new_c = F.normalize(new_c, dim=-1, eps=1e-12)
            shift = (new_c - centroids).pow(2).mean().sqrt().item()
            centroids = new_c
            if verbose:
                print(f"  [kmeans] iter {it+1}/{n_iter} shift={shift:.4e} "
                      f"used={int(mask.sum().item())}/{self.K}")

        # commit
        self.embedding.weight.data.copy_(centroids)
        if self.ema_update:
            # EMA buffer를 centroid로 시드하고, cluster_size를 1로 둬서
            # 학습 첫 step부터 EMA가 정상 동작하면서도 빠르게 반응하도록 함.
            self.ema_w.data.copy_(centroids)
            self.ema_cluster_size.data.fill_(1.0)

    # ── diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def codebook_usage(self, indices: torch.Tensor) -> float:
        """사용된 codebook entry 비율 반환."""
        used = indices.unique().numel()
        return used / self.K
