"""
utils/logging_utils.py

CSV + 콘솔 출력 기반 MetricLogger.
tensorboard / wandb 연동은 선택적으로 추가 가능.
"""

from __future__ import annotations
import os
import csv
import time
from collections import defaultdict
from typing import Any


class MetricLogger:
    """
    학습 중 발생하는 metric을 CSV에 누적 저장하고 콘솔에 출력.

    Usage:
        logger = MetricLogger("logs/tokenizer")
        logger.update(split="train", epoch=1, loss=0.42, perplexity=128.3)
        logger.update(split="val",   epoch=1, loss=0.38)
    """

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path       = os.path.join(log_dir, filename)
        self._fieldnames: list[str] = []
        self._file  = None
        self._writer = None
        self._start = time.time()
        self._step_count: dict[str, int] = defaultdict(int)

    # ── core ─────────────────────────────────────────────────────────────────

    def update(self, split: str, epoch: int, **kwargs: Any):
        row = {"split": split, "epoch": epoch,
               "elapsed": round(time.time() - self._start, 1),
               **kwargs}
        self._write_row(row)
        self._print(split, epoch, kwargs)

    def _write_row(self, row: dict):
        # 새 key 등장 시 파일 재생성 (헤더 추가)
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames += new_keys
            self._reopen()

        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()

    def _reopen(self):
        if self._file:
            self._file.close()
        self._file   = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def _print(self, split: str, epoch: int, metrics: dict):
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        print(f"  [{split}][ep{epoch:03d}] " + "  ".join(parts))

    def close(self):
        if self._file:
            self._file.close()

    def __del__(self):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Optional: wandb wrapper (wandb 설치 시 활성화)
# ──────────────────────────────────────────────────────────────────────────────

class WandbLogger:
    """
    MetricLogger와 동일한 인터페이스. wandb가 없으면 MetricLogger로 fallback.
    """

    def __init__(self, project: str, name: str, config: dict, log_dir: str):
        self._fallback = MetricLogger(log_dir)
        try:
            import wandb
            wandb.init(project=project, name=name, config=config)
            self._wandb = wandb
        except ImportError:
            print("[WandbLogger] wandb not installed, using CSV logger.")
            self._wandb = None

    def update(self, split: str, epoch: int, **kwargs):
        self._fallback.update(split, epoch, **kwargs)
        if self._wandb:
            self._wandb.log({f"{split}/{k}": v for k, v in kwargs.items()},
                            step=epoch)
