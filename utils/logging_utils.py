from __future__ import annotations
import os
import csv
import time
from collections import defaultdict
from typing import Any

class MetricLogger:
    def __init__(self, log_dir: str, filename: str = "metrics.csv",
                 verbose: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self.path       = os.path.join(log_dir, filename)
        self._fieldnames: list[str] = []
        self._file  = None
        self._writer = None
        self._start = time.time()
        self._step_count: dict[str, int] = defaultdict(int)
        self.verbose = verbose
    def update(self, split: str, epoch: int, **kwargs: Any):
        row = {"split": split, "epoch": epoch,
               "elapsed": round(time.time() - self._start, 1),
               **kwargs}
        self._write_row(row)
        if self.verbose:
            self._print(split, epoch, kwargs)
    def _write_row(self, row: dict):
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

class WandbLogger:
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
