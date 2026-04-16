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
from training.tokenizer.losses import total_vqvae_loss
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

    best_val_loss = float("inf")
    max_epochs    = cfg["training"]["max_epochs"]
    start_epoch   = 1
    global_step   = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    if resume is None:
        last_path = os.path.join(ckpt_dir, "last.pt")
        if os.path.exists(last_path):
            resume = last_path
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
        if is_main:
            print(f"[Resume] Loaded {resume}  → start_epoch={start_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}", flush=True)

    t_global = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "loss_rec": 0.0, "loss_vq": 0.0,
                   "loss_fid": 0.0, "perplexity": 0.0}
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
                alpha=loss_cfg["alpha"],
                beta=loss_cfg["beta"],
                use_gradient_loss=loss_cfg["use_gradient_loss"],
            )
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )
            opt.step()

            if is_main:
                vals = {
                    "loss":       losses["loss"].item(),
                    "loss_rec":   losses["loss_rec"].item(),
                    "loss_vq":    losses["loss_vq"].item(),
                    "loss_fid":   losses["loss_fid"].item(),
                    "perplexity": vq_dict["perplexity"].item(),
                }
                for k, v in vals.items():
                    running[k] += v
                n_steps += 1
                global_step += 1
                # CSV는 매 step 기록(상세 분석용), 콘솔은 tqdm postfix만
                logger.update(split="train", epoch=epoch, **vals)
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, global_step)
                if n_steps % 20 == 0:
                    pbar.set_postfix({
                        "loss": f"{running['loss']/n_steps:.3f}",
                        "rec":  f"{running['loss_rec']/n_steps:.3f}",
                        "ppl":  f"{running['perplexity']/n_steps:.1f}",
                    })
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
                f"ppl={avg['perplexity']:.2f}  "
                f"epoch_time={_fmt_dur(elapsed)}  "
                f"elapsed={_fmt_dur(total_elapsed)}  eta={_fmt_dur(eta)}",
                flush=True,
            )

        # ── eval ──────────────────────────────────────────────────────────
        if epoch % cfg["training"]["eval_every"] == 0:
            model.eval()
            local_sum, local_cnt = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["beat"].to(device, non_blocking=True)
                    x_hat, vq_dict = model(x)
                    losses = total_vqvae_loss(
                        x, x_hat, vq_dict["loss_vq"],
                        alpha=loss_cfg["alpha"],
                        beta=loss_cfg["beta"],
                        use_gradient_loss=loss_cfg["use_gradient_loss"],
                    )
                    local_sum += losses["loss"].item() * x.size(0)
                    local_cnt += x.size(0)

            stats = torch.tensor([local_sum, float(local_cnt)], device=device)
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            val_loss = (stats[0] / stats[1].clamp(min=1)).item()

            if is_main:
                logger.update(split="val", epoch=epoch, loss=val_loss)
                tb.add_scalar("val/loss", val_loss, epoch)
                tag = " ★best" if val_loss < best_val_loss else ""
                print(f"          val_loss={val_loss:.4f}{tag}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(raw_model, opt, epoch, val_loss,
                                    path=os.path.join(ckpt_dir, "best.pt"),
                                    model_cfg=cfg["model"])

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
