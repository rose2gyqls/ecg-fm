"""
Phase 1: VQ-VAE beat tokenizer training (DDP-aware).

Single GPU:
    python -m training.tokenizer.train --config configs/tokenizer/vqvae_base.yaml

Multi-GPU:
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
from collections import deque
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from models.tokenizer.vqvae import VQVAE
from training.tokenizer.losses import total_vqvae_loss, make_qrs_weight_map
from utils.checkpointing import save_checkpoint
from utils.logging_utils import MetricLogger


def setup_ddp():
    """Initialize DDP from torchrun env. Returns (False, 0, 1, 0) if not launched via torchrun."""
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


# ---------------------------------------------------------------------
# K-means warm-up: collect z_e via encoder forward, seed codebook via K-means.
# ---------------------------------------------------------------------
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
    Run encoder forward over a few batches to build a z_e pool, then call
    codebook.kmeans_init on it.

    DDP: each rank collects locally, sizes are equalized, all_gather builds
    the shared pool, rank 0 runs K-means, and the resulting buffers are
    broadcast back. After this returns, every rank holds the same codebook.
    """
    raw_model.eval()                       # eval BN -> z_e distribution stable
    K = raw_model.codebook.K
    D = raw_model.codebook.D

    per_rank = max(n_samples // max(world_size, 1), K * 8)

    local_chunks = []
    collected = 0
    pbar = tqdm(
        total=per_rank, desc="kmeans/collect", disable=(not is_main),
        leave=False, dynamic_ncols=True,
    )
    for batch in train_loader:
        x = batch["beat"].to(device, non_blocking=True)
        z = raw_model.encoder(x)
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

    local_pool = torch.cat(local_chunks, dim=0)[:per_rank]

    if ddp:
        # all_gather requires identical shapes -> trim to the smallest rank.
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

    if (not ddp) or rank == 0:
        raw_model.codebook.kmeans_init(pool, n_iter=n_iter, verbose=is_main)
    if ddp:
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

    # ---------- Data ----------
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
    pf = int(cfg["training"].get("prefetch_factor", 4))
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(pf if nw > 0 else None),
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

    # ---------- Model ----------
    # Persist data-pipeline contract on the checkpoint so downstream consumers
    # can self-describe (which normalization the tokenizer was trained with).
    cfg["model"].setdefault(
        "normalize", cfg["data"].get("normalize", "record_mad")
    )
    cfg["model"].setdefault(
        "record_mad_scale", float(cfg["data"].get("record_mad_scale", 5.0))
    )
    model = VQVAE(cfg["model"]).to(device)
    if is_main:
        print(f"[Tokenizer] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if ddp:
        # EMA buffers are already synchronized via all_reduce inside _ema_update,
        # so broadcast_buffers can stay False. find_unused_parameters is required
        # because embedding.weight gets EMA-updated outside the autograd graph.
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True,
        )
    raw_model = model.module if ddp else model

    # ---------- Optimizer / Scheduler ----------
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    # Real LR warmup. Earlier versions exposed `warmup_epochs` in config but
    # only constructed CosineAnnealingLR, so the field was a no-op. v4 wires
    # it through as a linear ramp from 1% lr → 100% lr over `warmup_epochs`,
    # then cosine-anneals over the remaining epochs. This eliminates the
    # epoch-1/2 transient where lr=3e-4 hits a freshly k-means-initialized
    # codebook and produces large early-step gradient noise.
    max_epochs    = cfg["training"]["max_epochs"]
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 0) or 0)
    if warmup_epochs > 0 and warmup_epochs < max_epochs:
        warmup = LinearLR(
            opt, start_factor=1e-2, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(opt, T_max=max_epochs - warmup_epochs)
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(opt, T_max=max_epochs)

    loss_cfg = cfg["training"]["loss"]
    ckpt_dir = cfg["training"]["ckpt_dir"]
    log_dir  = cfg["training"]["log_dir"]
    # TensorBoard dir: when set, multiple cb runs can share a parent directory
    # (e.g. logs/tb/tokenizer/cb{256,512,1024,2048}_v2) so a single
    # `tensorboard --logdir logs/tb/tokenizer` shows them all together.
    # Falls back to log_dir/tb for backwards compatibility.
    tb_dir   = cfg["training"].get("tb_dir") or os.path.join(log_dir, "tb")
    logger   = MetricLogger(log_dir) if is_main else None
    tb       = SummaryWriter(log_dir=tb_dir) if is_main else None
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[Tokenizer] log_dir={log_dir}  tb_dir={tb_dir}")

    # ---------- Loss configuration ----------
    use_grad_loss  = bool(loss_cfg.get("use_gradient_loss", True))
    use_fid_weight = bool(loss_cfg.get("fiducial_weight_map", False))
    alpha          = float(loss_cfg.get("alpha", 1.0))
    beta           = float(loss_cfg.get("beta", 0.5))
    gamma          = float(loss_cfg.get("gamma", 0.0))
    delta          = float(loss_cfg.get("delta", 0.0))     # entropy bonus
    spec_n_ffts    = tuple(loss_cfg.get("spec_n_ffts", (32, 64, 128)))

    # QRS weight map (only used when use_gradient_loss is False).
    fid_weights = None
    if (not use_grad_loss) and use_fid_weight:
        data_cfg = cfg.get("data", {})
        beat_length = int(data_cfg.get("beat_length", 256))
        before_ms = float(data_cfg.get("before_ms", 200))
        after_ms  = float(data_cfg.get("after_ms", 400))
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
            print(f"[Tokenizer] QRS weight map: r_pos={r_pos} "
                  f"sigma={loss_cfg.get('fid_sigma', 20.0)} "
                  f"peak={loss_cfg.get('fid_peak', 3.0)}")

    # Dead-code restart (0 disables).
    restart_every  = int(loss_cfg.get("restart_dead_every", 0) or 0)
    restart_thresh = float(loss_cfg.get("restart_dead_threshold", 1.0))

    # Early stopping (0 disables).
    es_patience = int(cfg["training"].get("early_stop_patience", 0) or 0)
    es_bad = 0

    # Sliding-window best-checkpoint selection. With high val_loss noise
    # (CV ~6% on this dataset) a single-eval `min` rewards lucky outliers
    # rather than a converged plateau — that's why v3's best.pt locked into
    # epoch 2/4 even though training kept improving the latent. Using the
    # mean over the last `best_window` evals smooths out per-epoch noise.
    # 1 disables (legacy behaviour: any single-eval improvement wins).
    best_window  = int(cfg["training"].get("best_window", 1) or 1)
    val_history: deque[float] = deque(maxlen=best_window)

    best_val_loss = float("inf")
    start_epoch   = 1
    global_step   = 0

    # ---------- Resume ----------
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

    # K-means warmup runs once before the first training step, only for fresh runs.
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

        # ---------- Train ----------
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "loss_rec": 0.0, "loss_vq": 0.0,
                   "loss_fid": 0.0, "loss_spec": 0.0, "loss_ent": 0.0,
                   "perplexity": 0.0, "n_dead_restarted": 0.0}
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
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                neg_entropy=vq_dict.get("neg_entropy"),
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

            # Dead-code restart: re-encode the same batch (no extra DataLoader
            # round-trip) and let the codebook swap dead entries.
            n_dead_restarted = 0
            if restart_every > 0 and (global_step + 1) % restart_every == 0:
                with torch.no_grad():
                    z_e = raw_model.encoder(x)
                n_dead_restarted = raw_model.codebook.restart_dead_codes(
                    z_e, threshold=restart_thresh,
                )

            global_step += 1
            if is_main:
                vals = {
                    "loss":       losses["loss"].item(),
                    "loss_rec":   losses["loss_rec"].item(),
                    "loss_vq":    losses["loss_vq"].item(),
                    "loss_fid":   losses["loss_fid"].item(),
                    "loss_spec":  losses["loss_spec"].item(),
                    "loss_ent":   losses["loss_ent"].item(),
                    "perplexity": vq_dict["perplexity"].item(),
                    "n_dead_restarted": float(n_dead_restarted),
                }
                for k, v in vals.items():
                    running[k] += v
                n_steps += 1
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
        pbar.close()
        scheduler.step()

        # ---------- Epoch summary (rank 0) ----------
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
                f"spec={avg['loss_spec']:.4f}  ent={avg['loss_ent']:.4f}  "
                f"ppl={avg['perplexity']:.2f}  "
                f"dead_restart_avg={avg['n_dead_restarted']:.2f}  "
                f"epoch_time={_fmt_dur(elapsed)}  "
                f"elapsed={_fmt_dur(total_elapsed)}  eta={_fmt_dur(eta)}",
                flush=True,
            )

        # ---------- Eval ----------
        do_eval = (epoch % cfg["training"]["eval_every"] == 0)
        val_loss_for_es = None
        if do_eval:
            model.eval()
            local_sums = {"loss": 0.0, "loss_rec": 0.0, "loss_vq": 0.0,
                          "loss_fid": 0.0, "loss_spec": 0.0, "loss_ent": 0.0,
                          "perplexity": 0.0}
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
                        delta=delta,
                        neg_entropy=vq_dict.get("neg_entropy"),
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
                    local_sums["loss_ent"]  += losses["loss_ent"].item() * bs
                    local_sums["perplexity"] += vq_dict["perplexity"].item() * bs
                    local_cnt += bs

            keys = ["loss", "loss_rec", "loss_vq", "loss_fid",
                    "loss_spec", "loss_ent", "perplexity"]
            stats = torch.tensor(
                [local_sums[k] for k in keys] + [float(local_cnt)],
                device=device,
            )
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            cnt = stats[-1].clamp(min=1)
            val_metrics = {k: (stats[i] / cnt).item() for i, k in enumerate(keys)}
            val_loss = val_metrics["loss"]

            # Smoothed val_loss for both best.pt selection and early stop.
            # Falls back to raw val_loss if best_window <= 1.
            val_history.append(val_loss)
            val_loss_smoothed = sum(val_history) / len(val_history)
            val_loss_for_es = val_loss_smoothed

            if is_main:
                logger.update(split="val", epoch=epoch, **val_metrics)
                for k, v in val_metrics.items():
                    tb.add_scalar(f"val/{k}", v, epoch)
                tb.add_scalar("val/loss_smoothed", val_loss_smoothed, epoch)
                improved = val_loss_smoothed < best_val_loss
                tag = " ★best" if improved else ""
                print(
                    f"          val  loss={val_loss:.4f}  "
                    f"smooth={val_loss_smoothed:.4f}  "
                    f"rec={val_metrics['loss_rec']:.4f}  "
                    f"vq={val_metrics['loss_vq']:.4f}  "
                    f"fid={val_metrics['loss_fid']:.4f}  "
                    f"spec={val_metrics['loss_spec']:.4f}  "
                    f"ent={val_metrics['loss_ent']:.4f}  "
                    f"ppl={val_metrics['perplexity']:.1f}{tag}",
                    flush=True,
                )

                if improved:
                    best_val_loss = val_loss_smoothed
                    es_bad = 0
                    save_checkpoint(raw_model, opt, epoch, val_loss_smoothed,
                                    path=os.path.join(ckpt_dir, "best.pt"),
                                    model_cfg=cfg["model"])
                else:
                    es_bad += 1

        # ---------- Periodic + last checkpoint ----------
        if epoch % cfg["training"]["save_every"] == 0 and is_main:
            save_checkpoint(raw_model, opt, epoch, None,
                            path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                            model_cfg=cfg["model"])

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

        # ---------- Early stopping (broadcast across ranks) ----------
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
                        help="Checkpoint path to resume from. "
                             "If omitted, ckpt_dir/last.pt is auto-loaded if present.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
