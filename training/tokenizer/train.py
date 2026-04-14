"""
training/tokenizer/train.py

Phase 1: VQ-VAE Beat Tokenizer 학습
Usage:
    python -m training.tokenizer.train --config configs/tokenizer/vqvae_base.yaml
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.tokenizer.vqvae import VQVAE
from training.tokenizer.losses import total_vqvae_loss
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging_utils import MetricLogger


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tokenizer] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    source = cfg["data"].get("source", "npy")
    if source == "heedb":
        from data.datasets.heedb_beat_dataset import HEEDBBeatDataset as _DS
    else:
        from data.datasets.beat_dataset import BeatDataset as _DS
    train_ds = _DS(cfg["data"], split="train")
    val_ds   = _DS(cfg["data"], split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = VQVAE(cfg["model"]).to(device)
    print(f"[Tokenizer] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    opt = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(opt, T_max=cfg["training"]["max_epochs"])

    loss_cfg = cfg["training"]["loss"]
    logger   = MetricLogger(cfg["training"]["log_dir"])
    ckpt_dir = cfg["training"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            x = batch["beat"].to(device)          # (B, 1, W)
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

            logger.update(
                split="train", epoch=epoch,
                loss=losses["loss"].item(),
                loss_rec=losses["loss_rec"].item(),
                loss_vq=losses["loss_vq"].item(),
                loss_fid=losses["loss_fid"].item(),
                perplexity=vq_dict["perplexity"].item(),
            )

        scheduler.step()

        # ── eval ──────────────────────────────────────────────────────────
        if epoch % cfg["training"]["eval_every"] == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["beat"].to(device)
                    x_hat, vq_dict = model(x)
                    losses = total_vqvae_loss(
                        x, x_hat, vq_dict["loss_vq"],
                        alpha=loss_cfg["alpha"],
                        beta=loss_cfg["beta"],
                        use_gradient_loss=loss_cfg["use_gradient_loss"],
                    )
                    val_losses.append(losses["loss"].item())

            val_loss = sum(val_losses) / len(val_losses)
            logger.update(split="val", epoch=epoch, loss=val_loss)
            print(f"[Epoch {epoch:03d}] val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, opt, epoch, val_loss,
                                path=os.path.join(ckpt_dir, "best.pt"),
                                model_cfg=cfg["model"])

        # ── periodic save ─────────────────────────────────────────────────
        if epoch % cfg["training"]["save_every"] == 0:
            save_checkpoint(model, opt, epoch, None,
                            path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                            model_cfg=cfg["model"])

    print(f"[Tokenizer] Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer/vqvae_base.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
