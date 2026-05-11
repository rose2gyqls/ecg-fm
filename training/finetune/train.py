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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from models.tokenizer.vqvae import VQVAE
from models.transformer.ecg_model import ECGFoundationModel
from models.heads.mlm_head import ClassifierHead
from utils.checkpointing import save_checkpoint
from utils.logging_utils import MetricLogger

def build_optimizer(model, head, cfg):
    ft_cfg = cfg["training"]
    freeze_n = cfg["model"]["classifier"].get("freeze_layers", 0)
    if cfg["model"].get("freeze_transformer", False):
        for p in model.parameters():
            p.requires_grad_(False)
        params = list(head.parameters())
    elif freeze_n > 0:
        for i, layer in enumerate(model.transformer.layers):
            if i < freeze_n:
                for p in layer.parameters():
                    p.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    else:
        params = list(model.parameters()) + list(head.parameters())
    return AdamW(params, lr=ft_cfg["lr"], weight_decay=ft_cfg["weight_decay"]), params

def evaluate(model, tokenizer, head, loader, device, n_classes):
    model.eval()
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            beats    = batch["beats"].to(device)
            rr_feats = batch["rr_feats"].to(device)
            stft     = batch["stft"].to(device)
            labels   = batch["label"].to(device)
            B, N, L, W = beats.shape
            _, idx_flat = tokenizer.encode(beats.view(B*N*L, 1, W))
            indices = idx_flat.view(B, N, L)
            out     = model(indices, rr_feats, stft)
            logits  = head(out)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(-1).numpy()
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auroc = (
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            if n_classes > 2
            else roc_auc_score(labels, probs[:, 1])
        )
    except Exception:
        auroc = 0.0
    return {"acc": acc, "f1": f1, "auroc": auroc}

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Finetune] Device: {device}")
    tok_ckpt = torch.load(cfg.get("tokenizer_ckpt",
                                   "checkpoints/tokenizer/best.pt"),
                          map_location="cpu")
    tokenizer = VQVAE(tok_ckpt.get("model_cfg", {}))
    tokenizer.load_state_dict(tok_ckpt["model"])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    pt_ckpt = torch.load(cfg["base_pretrain_ckpt"], map_location="cpu")
    model = ECGFoundationModel(pt_ckpt.get("model_cfg", cfg["model"])).to(device)
    model.load_state_dict(pt_ckpt["model"], strict=False)
    head = ClassifierHead(
        d_model=cfg["model"]["d_model"],
        n_classes=cfg["data"]["n_classes"],
        **cfg["model"]["classifier"],
    ).to(device)
    opt, params = build_optimizer(model, head, cfg)
    scheduler = CosineAnnealingLR(opt, T_max=cfg["training"]["max_epochs"])
    from data.datasets.finetune_dataset import FinetuneDataset
    train_ds = FinetuneDataset(cfg["data"], split="train")
    val_ds   = FinetuneDataset(cfg["data"], split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=cfg["training"]["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["training"]["num_workers"])
    logger   = MetricLogger(cfg["training"]["log_dir"])
    ckpt_dir = cfg["training"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auroc = 0.0
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        head.train()
        for batch in train_loader:
            beats    = batch["beats"].to(device)
            rr_feats = batch["rr_feats"].to(device)
            stft     = batch["stft"].to(device)
            labels   = batch["label"].to(device)
            B, N, L, W = beats.shape
            with torch.no_grad():
                _, idx_flat = tokenizer.encode(beats.view(B*N*L, 1, W))
            indices = idx_flat.view(B, N, L)
            out    = model(indices, rr_feats, stft)
            logits = head(out)
            loss   = F.cross_entropy(logits, labels,
                                     label_smoothing=cfg["training"].get(
                                         "label_smoothing", 0.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip"])
            opt.step()
            logger.update(split="train", epoch=epoch, loss=loss.item())
        scheduler.step()
        metrics = evaluate(model, tokenizer, head, val_loader, device,
                           cfg["data"]["n_classes"])
        logger.update(split="val", epoch=epoch, **metrics)
        print(f"[Epoch {epoch:03d}] "
              f"auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f} "
              f"acc={metrics['acc']:.4f}")
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            save_checkpoint(model, opt, epoch, metrics["auroc"],
                            path=os.path.join(ckpt_dir, "best.pt"),
                            extra={"head": head.state_dict()})
    print(f"[Finetune] Done. Best AUROC={best_auroc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune/arrhythmia.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
