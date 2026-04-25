"""
training/tokenizer/train.py

Phase 1: VQ-VAE Beat Tokenizer 학습 (DDP 지원)

Single GPU:
    python -m training.tokenizer.train --config configs/tokenizer/vqvae_base.yaml

Multi-GPU (예: GPU 0,1):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        -m training.tokenizer.train --config configs/tokenizer/vqvae_heedb.yaml
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.tokenizer.vqvae import VQVAE
from training.tokenizer.losses import total_vqvae_loss, make_qrs_weight_map
from utils.checkpointing import save_checkpoint
from utils.logging_utils import MetricLogger


def setup_ddp():
    """torchrun에서 주입된 env로 DDP 초기화. 단일 프로세스면 (False, 0, 1, 0)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# A-3: K-means warm-up — encoder만 forward해서 z_e 풀 모은 뒤 codebook 시드
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kmeans_warmup(
    raw_model: VQVAE,
    train_loader: DataLoader,
    device: torch.device,
    ddp: bool,
    rank: int,
    world_size: int,
    n_samples: int = 50_000,
    n_iter: int = 10,
    is_main: bool = True,
):
    """
    raw_model.encoder만 forward해서 z_e 풀을 구성하고, codebook을 K-means로 초기화.

    DDP: 각 rank가 로컬 batch에서 z_e 모아 → all_gather로 풀 합치기 →
         rank 0에서 K-means → 결과를 broadcast.
    """
    raw_model.eval()  # BN을 eval로 둬야 z_e 분포가 일관됨
    K = raw_model.codebook.K
    D = raw_model.codebook.D

    # 각 rank가 모을 로컬 표본 수 (균등 분할). batch 단위로 채우다 보면 약간 더 걸림.
    per_rank = max(n_samples // max(world_size, 1), K * 8)

    local_chunks = []
    collected = 0
    pbar = tqdm(
        total=per_rank, desc="kmeans/collect", disable=(not is_main),
        leave=False, dynamic_ncols=True,
    )
    for batch in train_loader:
        x = batch["beat"].to(device, non_blocking=True)
        z = raw_model.encoder(x)              # (B, D)
        local_chunks.append(z.detach())
        collected += z.shape[0]
        pbar.update(z.shape[0])
        if collected >= per_rank:
            break
    pbar.close()

    if not local_chunks:
        if is_main:
            print("[kmeans_warmup] no samples collected; skipping.")
        return

    local_pool = torch.cat(local_chunks, dim=0)[:per_rank]  # (Pl, D)

    # 모든 rank가 동일 크기로 맞춰야 all_gather 가능 → 가장 작은 rank 길이로 잘라냄.
    if ddp:
        sizes = [torch.tensor([0], device=device) for _ in range(world_size)]
        dist.all_gather(sizes, torch.tensor([local_pool.shape[0]], device=device))
        min_size = int(min(s.item() for s in sizes))
        local_pool = local_pool[:min_size].contiguous()
        gathered = [torch.zeros_like(local_pool) for _ in range(world_size)]
        dist.all_gather(gathered, local_pool)
        pool = torch.cat(gathered, dim=0)
    else:
        pool = local_pool

    if is_main:
        print(f"[kmeans_warmup] pool shape = {tuple(pool.shape)} (K={K}, D={D})")

    # rank 0에서 K-means → 결과를 모든 rank로 broadcast
    if (not ddp) or rank == 0:
        raw_model.codebook.kmeans_init(pool, n_iter=n_iter, verbose=is_main)
    if ddp:
        # codebook.embedding.weight + ema buffers를 broadcast
        dist.broadcast(raw_model.codebook.embedding.weight.data, src=0)
        if raw_model.codebook.ema_update:
            dist.broadcast(raw_model.codebook.ema_w.data, src=0)
            dist.broadcast(raw_model.codebook.ema_cluster_size.data, src=0)


def train(cfg: dict, resume: str | None = None):
    ddp, rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main:
        print(f"[Tokenizer] DDP={ddp}  world_size={world_size}  device={device}")

    # ── Data ────────────────────────────────────────────────────────────────
    source = cfg["data"].get("source", "npy")
    if source == "heedb":
        from data.datasets.heedb_beat_dataset import HEEDBBeatDataset as _DS
    else:
        from data.datasets.beat_dataset import BeatDataset as _DS
    train_ds = _DS(cfg["data"], split="train")
    val_ds   = _DS(cfg["data"], split="val")

    if ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(cfg["data"].get("seed", 42)),
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler   = None

    nw = int(cfg["training"]["num_workers"])
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = VQVAE(cfg["model"]).to(device)
    if is_main:
        print(f"[Tokenizer] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if ddp:
        # EMA 버퍼는 내부에서 all_reduce로 이미 동기화됨 → broadcast_buffers 불필요.
        # EMA codebook의 embedding.weight는 autograd가 아닌 EMA로 갱신되므로
        # backward 그래프에 나타나지 않음 → find_unused_parameters=True 필요.
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True,
        )
    raw_model = model.module if ddp else model

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(opt, T_max=cfg["training"]["max_epochs"])

    loss_cfg = cfg["training"]["loss"]
    ckpt_dir = cfg["training"]["ckpt_dir"]
    log_dir  = cfg["training"]["log_dir"]
    logger   = MetricLogger(log_dir) if is_main else None
    tb       = SummaryWriter(log_dir=os.path.join(log_dir, "tb")) if is_main else None
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Loss configuration ───────────────────────────────────────────────────
    use_grad_loss     = bool(loss_cfg.get("use_gradient_loss", True))
    use_fid_weight    = bool(loss_cfg.get("fiducial_weight_map", False))
    alpha             = float(loss_cfg.get("alpha", 1.0))
    beta              = float(loss_cfg.get("beta", 0.5))
    gamma             = float(loss_cfg.get("gamma", 0.0))
    spec_n_ffts       = tuple(loss_cfg.get("spec_n_ffts", (32, 64, 128)))

    # QRS 가중치 맵 — fiducial_weight_map 모드에서만 사용
    fid_weights = None
    if (not use_grad_loss) and use_fid_weight:
        # 기본 geometry: before_ms=200, after_ms=400 → R at 1/3 of beat_length
        beat_length = int(cfg.get("data", {}).get("beat_length", 256))
        before_ms = float(cfg.get("data", {}).get("before_ms", 200))
        after_ms = float(cfg.get("data", {}).get("after_ms", 400))
        r_pos = int(round(beat_length * before_ms / (before_ms + after_ms)))
        fid_weights = make_qrs_weight_map(
            beat_length=beat_length,
            r_pos=r_pos,
            sigma=float(loss_cfg.get("fid_sigma", 20.0)),
            base_weight=float(loss_cfg.get("fid_base", 1.0)),
            peak_weight=float(loss_cfg.get("fid_peak", 3.0)),
            device=device,
        )
        if is_main:
            print(f"[Tokenizer] QRS weight map: r_pos={r_pos} sigma={loss_cfg.get('fid_sigma', 20.0)} "
                  f"peak={loss_cfg.get('fid_peak', 3.0)}")

    # ── A-2: dead-code restart settings ──────────────────────────────────────
    restart_every = int(loss_cfg.get("restart_dead_every", 0) or 0)   # 0 = disabled
    restart_thresh = float(loss_cfg.get("restart_dead_threshold", 1.0))

    # ── A-7: early stopping settings ─────────────────────────────────────────
    es_patience = int(cfg["training"].get("early_stop_patience", 0) or 0)  # 0 = disabled
    es_bad = 0

    best_val_loss = float("inf")
    max_epochs    = cfg["training"]["max_epochs"]
    start_epoch   = 1
    global_step   = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    if resume is None:
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_path):
            resume = last_path
    resumed_from_ckpt = False
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("metric") or float("inf")))
        global_step   = int(ckpt.get("global_step", 0))
        resumed_from_ckpt = True
        if is_main:
            print(f"[Resume] Loaded {resume}  → start_epoch={start_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}", flush=True)

    # ── A-3: K-means warmup (resume 아닐 때만 1회 수행) ─────────────────────
    kmeans_cfg = cfg["training"].get("kmeans_init", {}) or {}
    if (
        not resumed_from_ckpt
        and bool(kmeans_cfg.get("enabled", False))
        and raw_model.codebook.ema_update
    ):
        if is_main:
            print(f"[Tokenizer] K-means warmup: "
                  f"n_samples={kmeans_cfg.get('n_samples', 50_000)} "
                  f"n_iter={kmeans_cfg.get('n_iter', 10)}", flush=True)
        # 별도 sampler/loader: 가능한 한 매 rank가 다른 batch를 보도록 set_epoch
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(0)
        kmeans_warmup(
            raw_model, train_loader, device, ddp, rank, world_size,
            n_samples=int(kmeans_cfg.get("n_samples", 50_000)),
            n_iter=int(kmeans_cfg.get("n_iter", 10)),
            is_main=is_main,
        )
        if ddp:
            dist.barrier()

    t_global = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "loss_rec": 0.0, "loss_vq": 0.0,
                   "loss_fid": 0.0, "loss_spec": 0.0, "perplexity": 0.0,
                   "n_dead_restarted": 0.0}
        n_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"ep{epoch:03d}/{max_epochs:03d}",
            disable=(not is_main),
            dynamic_ncols=True, mininterval=1.0, leave=False,
        )
        for batch in pbar:
            x = batch["beat"].to(device, non_blocking=True)       # (B, 1, W)
            x_hat, vq_dict = model(x)
            losses = total_vqvae_loss(
                x, x_hat, vq_dict["loss_vq"],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                use_gradient_loss=use_grad_loss,
                fiducial_weights=fid_weights,
                spec_n_ffts=spec_n_ffts,
            )
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )
            opt.step()

            # ── A-2: dead-code restart (post-step, every N steps) ──────────
            n_dead_restarted = 0
            if restart_every > 0 and (global_step + 1) % restart_every == 0:
                # encoder forward로 다시 z_e 뽑지 않고, autograd 그래프 밖에서 재사용
                with torch.no_grad():
                    z_e = raw_model.encoder(x)
                n_dead_restarted = raw_model.codebook.restart_dead_codes(
                    z_e, threshold=restart_thresh,
                )

            if is_main:
                vals = {
                    "loss":       losses["loss"].item(),
                    "loss_rec":   losses["loss_rec"].item(),
                    "loss_vq":    losses["loss_vq"].item(),
                    "loss_fid":   losses["loss_fid"].item(),
                    "loss_spec":  losses["loss_spec"].item(),
                    "perplexity": vq_dict["perplexity"].item(),
                    "n_dead_restarted": float(n_dead_restarted),
                }
                for k, v in vals.items():
                    running[k] += v
                n_steps += 1
                global_step += 1
                logger.update(split="train", epoch=epoch, **vals)
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, global_step)
                if n_steps % 20 == 0:
                    pbar.set_postfix({
                        "loss": f"{running['loss']/n_steps:.3f}",
                        "rec":  f"{running['loss_rec']/n_steps:.3f}",
                        "vq":   f"{running['loss_vq']/n_steps:.3f}",
                        "ppl":  f"{running['perplexity']/n_steps:.1f}",
                    })
            else:
                # rank>0도 step 카운터 증가 (restart 주기 일치를 위해)
                global_step += 1
        pbar.close()
        scheduler.step()

        # ── epoch summary (rank 0) ────────────────────────────────────────
        if is_main and n_steps > 0:
            avg = {k: v / n_steps for k, v in running.items()}
            for k, v in avg.items():
                tb.add_scalar(f"train_epoch/{k}", v, epoch)
            tb.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            elapsed = time.time() - t_epoch
            total_elapsed = time.time() - t_global
            eta = elapsed * (max_epochs - epoch)
            print(
                f"[ep{epoch:03d}/{max_epochs:03d}] "
                f"loss={avg['loss']:.4f}  rec={avg['loss_rec']:.4f}  "
                f"vq={avg['loss_vq']:.4f}  fid={avg['loss_fid']:.4f}  "
                f"spec={avg['loss_spec']:.4f}  "
                f"ppl={avg['perplexity']:.2f}  "
                f"dead_restart_avg={avg['n_dead_restarted']:.2f}  "
                f"epoch_time={_fmt_dur(elapsed)}  "
                f"elapsed={_fmt_dur(total_elapsed)}  eta={_fmt_dur(eta)}",
                flush=True,
            )

        # ── eval ──────────────────────────────────────────────────────────
        do_eval = (epoch % cfg["training"]["eval_every"] == 0)
        val_loss_for_es = None
        if do_eval:
            model.eval()
            local_sums = {"loss": 0.0, "loss_rec": 0.0, "loss_vq": 0.0,
                          "loss_fid": 0.0, "loss_spec": 0.0, "perplexity": 0.0}
            local_cnt = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["beat"].to(device, non_blocking=True)
                    x_hat, vq_dict = model(x)
                    losses = total_vqvae_loss(
                        x, x_hat, vq_dict["loss_vq"],
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        use_gradient_loss=use_grad_loss,
                        fiducial_weights=fid_weights,
                        spec_n_ffts=spec_n_ffts,
                    )
                    bs = x.size(0)
                    local_sums["loss"]      += losses["loss"].item() * bs
                    local_sums["loss_rec"]  += losses["loss_rec"].item() * bs
                    local_sums["loss_vq"]   += losses["loss_vq"].item() * bs
                    local_sums["loss_fid"]  += losses["loss_fid"].item() * bs
                    local_sums["loss_spec"] += losses["loss_spec"].item() * bs
                    local_sums["perplexity"] += vq_dict["perplexity"].item() * bs
                    local_cnt += bs

            keys = ["loss", "loss_rec", "loss_vq", "loss_fid", "loss_spec", "perplexity"]
            stats = torch.tensor(
                [local_sums[k] for k in keys] + [float(local_cnt)],
                device=device,
            )
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            cnt = stats[-1].clamp(min=1)
            val_metrics = {k: (stats[i] / cnt).item() for i, k in enumerate(keys)}
            val_loss = val_metrics["loss"]
            val_loss_for_es = val_loss

            if is_main:
                logger.update(split="val", epoch=epoch, **val_metrics)
                for k, v in val_metrics.items():
                    tb.add_scalar(f"val/{k}", v, epoch)
                tag = " ★best" if val_loss < best_val_loss else ""
                print(
                    f"          val  loss={val_loss:.4f}  "
                    f"rec={val_metrics['loss_rec']:.4f}  "
                    f"vq={val_metrics['loss_vq']:.4f}  "
                    f"fid={val_metrics['loss_fid']:.4f}  "
                    f"spec={val_metrics['loss_spec']:.4f}  "
                    f"ppl={val_metrics['perplexity']:.1f}{tag}",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    es_bad = 0
                    save_checkpoint(raw_model, opt, epoch, val_loss,
                                    path=os.path.join(ckpt_dir, "best.pt"),
                                    model_cfg=cfg["model"])
                else:
                    es_bad += 1

        # ── periodic save ─────────────────────────────────────────────────
        if epoch % cfg["training"]["save_every"] == 0 and is_main:
            save_checkpoint(raw_model, opt, epoch, None,
                            path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                            model_cfg=cfg["model"])

        # ── always save last.pt (for resume) ──────────────────────────────
        if is_main:
            save_checkpoint(
                raw_model, opt, epoch, best_val_loss,
                path=os.path.join(ckpt_dir, "last.pt"),
                model_cfg=cfg["model"],
                extra={
                    "scheduler":     scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "global_step":   global_step,
                },
            )

        if ddp:
            dist.barrier()

        # ── A-7: early stopping (after eval; broadcast across ranks) ──────
        if es_patience > 0 and val_loss_for_es is not None:
            stop_flag = 0
            if is_main and es_bad >= es_patience:
                stop_flag = 1
                print(f"[EarlyStop] no val_loss improvement for {es_patience} evals; "
                      f"stopping at epoch {epoch}.", flush=True)
            if ddp:
                t = torch.tensor(stop_flag, device=device)
                dist.broadcast(t, src=0)
                stop_flag = int(t.item())
            if stop_flag:
                break

    if is_main:
        print(f"[Tokenizer] Training complete. Best val_loss={best_val_loss:.4f}")
        if tb is not None:
            tb.close()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer/vqvae_base.yaml")
    parser.add_argument("--resume", default=None,
                        help="ckpt path to resume from. 미지정 시 ckpt_dir/last.pt 자동 로드.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
