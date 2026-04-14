"""
training/pretrain/train.py

Phase 3: Masked Beat Modeling pre-training
Usage:
    python -m training.pretrain.train --config configs/pretrain/masked_beat_base.yaml
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.tokenizer.vqvae import VQVAE
from models.transformer.ecg_model import ECGFoundationModel
from models.heads.mlm_head import MaskedBeatModelingHead, MaskedRhythmHead
from training.pretrain.masking import apply_masking
from utils.checkpointing import save_checkpoint
from utils.logging_utils import MetricLogger


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Device: {device}")

    # ── Frozen tokenizer ────────────────────────────────────────────────────
    ckpt = torch.load(cfg["tokenizer"]["ckpt"], map_location="cpu")
    tok_model_cfg = _load_tok_cfg(cfg, ckpt)
    tokenizer = VQVAE(tok_model_cfg)
    tokenizer.load_state_dict(ckpt["model"])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    print("[Pretrain] Tokenizer loaded and frozen.")

    # ── ECG-FM + heads ──────────────────────────────────────────────────────
    model    = ECGFoundationModel(cfg["model"]).to(device)
    mlm_head = MaskedBeatModelingHead(
        d_model=cfg["model"]["d_model"],
        codebook_size=cfg["tokenizer"]["codebook_size"],
    ).to(device)
    rr_head  = MaskedRhythmHead(d_model=cfg["model"]["d_model"]).to(device)

    params = (list(model.parameters()) +
              list(mlm_head.parameters()) +
              list(rr_head.parameters()))
    print(f"[Pretrain] Parameters: {sum(p.numel() for p in params):,}")

    # ── Data ────────────────────────────────────────────────────────────────
    source = cfg["data"].get("source", "npy")
    if source == "heedb":
        from data.datasets.heedb_ecg_dataset import HEEDBECGDataset as _DS
    else:
        from data.datasets.ecg_dataset import ECGDataset as _DS
    train_ds = _DS(cfg["data"], split="train")
    val_ds   = _DS(cfg["data"], split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["training"]["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["training"]["num_workers"])

    # ── Optimizer ───────────────────────────────────────────────────────────
    opt = AdamW(params, lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(opt, T_max=cfg["training"]["max_epochs"])
    mask_cfg  = cfg["masking"]
    loss_cfg  = cfg["training"]["loss"]
    logger    = MetricLogger(cfg["training"]["log_dir"])
    ckpt_dir  = cfg["training"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    mask_token_id = cfg["tokenizer"]["codebook_size"]   # MASK token id
    best_val = float("inf")

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train(); mlm_head.train(); rr_head.train()

        for batch in train_loader:
            # batch keys: beats (B,N,12,W), rr_feats (B,N,12,3), stft (B,12,F,T')
            beats    = batch["beats"].to(device)      # (B, N, 12, W)
            rr_feats = batch["rr_feats"].to(device)   # (B, N, 12, 3)
            stft     = batch["stft"].to(device)       # (B, 12, F, T')

            # ── tokenize beats with frozen VQ-VAE ─────────────────────────
            B, N, L, W = beats.shape
            beats_flat = beats.view(B * N * L, 1, W)
            with torch.no_grad():
                _, indices_flat = tokenizer.encode(beats_flat)
            indices = indices_flat.view(B, N, L)      # (B, N, 12)

            # ── masking ───────────────────────────────────────────────────
            masked = apply_masking(
                indices, rr_feats,
                beat_mask_ratio=mask_cfg["beat_mask_ratio"],
                rhythm_mask_ratio=mask_cfg["rhythm_mask_ratio"],
                lead_dropout_prob=mask_cfg["lead_dropout_prob"],
                lead_min_leads=mask_cfg["lead_dropout_min_leads"],
                mask_token_id=mask_token_id,
            )

            # ── forward ───────────────────────────────────────────────────
            out = model(
                masked["masked_indices"],
                masked["masked_rr_feats"],
                stft,
            )  # (B, 1+N*L, d)

            # token positions (skip CLS token at position 0)
            token_out = out[:, 1:, :].view(B, N, L, -1)   # (B, N, 12, d)

            # ── MLM loss: CE on masked beat tokens ────────────────────────
            beat_mask = masked["beat_mask"]                 # (B, N, 12)
            if beat_mask.any():
                logits_mlm = mlm_head(token_out[beat_mask]) # (M, K)
                targets    = indices[beat_mask]              # (M,)
                loss_mlm   = F.cross_entropy(logits_mlm, targets)
            else:
                loss_mlm = torch.tensor(0.0, device=device)

            # ── Rhythm loss: MSE on masked RR ─────────────────────────────
            rr_mask = masked["rhythm_mask"]                 # (B, N, 12)
            if rr_mask.any():
                pred_rr  = rr_head(token_out[rr_mask])      # (M, 3)
                true_rr  = rr_feats[rr_mask]                # (M, 3)
                loss_rr  = F.mse_loss(pred_rr, true_rr)
            else:
                loss_rr = torch.tensor(0.0, device=device)

            loss = (loss_cfg["morphology_weight"] * loss_mlm
                    + loss_cfg["rhythm_weight"]    * loss_rr)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip"])
            opt.step()

            logger.update(split="train", epoch=epoch,
                          loss=loss.item(),
                          loss_mlm=loss_mlm.item(),
                          loss_rr=loss_rr.item())

        scheduler.step()

        if epoch % cfg["training"]["eval_every"] == 0:
            # minimal val loop
            model.eval(); mlm_head.eval(); rr_head.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    beats    = batch["beats"].to(device)
                    rr_feats = batch["rr_feats"].to(device)
                    stft     = batch["stft"].to(device)
                    B, N, L, W = beats.shape
                    _, indices_flat = tokenizer.encode(beats.view(B*N*L,1,W))
                    indices = indices_flat.view(B, N, L)
                    masked = apply_masking(indices, rr_feats,
                                          mask_token_id=mask_token_id)
                    out = model(masked["masked_indices"],
                                masked["masked_rr_feats"], stft)
                    token_out = out[:, 1:, :].view(B, N, L, -1)
                    bm = masked["beat_mask"]
                    if bm.any():
                        l = F.cross_entropy(mlm_head(token_out[bm]),
                                            indices[bm])
                        val_losses.append(l.item())

            if val_losses:
                vl = sum(val_losses) / len(val_losses)
                logger.update(split="val", epoch=epoch, loss=vl)
                print(f"[Epoch {epoch:03d}] val_mlm_loss={vl:.4f}")
                if vl < best_val:
                    best_val = vl
                    save_checkpoint(model, opt, epoch, vl,
                                    path=os.path.join(ckpt_dir, "best.pt"),
                                    extra={"mlm_head": mlm_head.state_dict(),
                                           "rr_head": rr_head.state_dict()})

        if epoch % cfg["training"]["save_every"] == 0:
            save_checkpoint(model, opt, epoch, None,
                            path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    print(f"[Pretrain] Done. Best val loss={best_val:.4f}")


def _load_tok_cfg(cfg, ckpt=None):
    """
    tokenizer model cfg 우선순위:
      1) ckpt["model_cfg"]  (tokenizer 학습 시 save_checkpoint가 동봉한 경우)
      2) cfg["tokenizer"]["model_cfg_yaml"] 경로의 YAML
      3) cfg["tokenizer"]["model"] 딕셔너리
    """
    if ckpt is None:
        ckpt = torch.load(cfg["tokenizer"]["ckpt"], map_location="cpu")
    if "model_cfg" in ckpt and ckpt["model_cfg"]:
        return ckpt["model_cfg"]
    tok = cfg.get("tokenizer", {})
    if tok.get("model_cfg_yaml"):
        with open(tok["model_cfg_yaml"]) as f:
            return yaml.safe_load(f)["model"]
    if "model" in tok:
        return tok["model"]
    raise ValueError("Tokenizer model cfg not found: save ckpt with model_cfg or "
                     "set tokenizer.model_cfg_yaml / tokenizer.model in config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain/masked_beat_base.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
