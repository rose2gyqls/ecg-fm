from __future__ import annotations
import os
import torch
from typing import Any, Optional

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: Optional[float],
    path: str,
    model_cfg: Optional[dict] = None,
    extra: Optional[dict] = None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch":     epoch,
        "metric":    metric,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_cfg": model_cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Checkpoint] Saved -> {path}  (epoch={epoch}, metric={metric})")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded <- {path}  (epoch={ckpt.get('epoch')}, "
          f"metric={ckpt.get('metric')})")
    return ckpt

def load_model_only(
    path: str,
    model: torch.nn.Module,
    strict: bool = False,
    device: str = "cpu",
) -> None:
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
