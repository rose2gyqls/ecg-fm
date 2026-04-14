"""
data/datasets/finetune_dataset.py

Phase 4 Fine-tuning용 Dataset.
ECGDataset을 상속하고 label을 추가로 반환.

파일 포맷 (npy dict):
  {
    "signal" : (12, T)
    "label"  : int or str
    "rpeaks" : (N,)   (선택)
  }

h5 포맷:
  hf["signal"] : (12, T)
  hf["label"]  : scalar int
"""

from __future__ import annotations
import os
import glob
import json
import numpy as np
import torch

from data.datasets.ecg_dataset import ECGDataset


class FinetuneDataset(ECGDataset):
    """
    ECGDataset에 label 필드 추가.

    label_map (선택): str label -> int index 매핑 dict.
    cfg에 label_map_path를 지정하거나, label이 이미 int면 그대로 사용.
    """

    def __init__(self, cfg: dict, split: str = "train"):
        super().__init__(cfg, split)
        self.label_col = cfg.get("label_col", "label")

        # optional label -> int mapping
        lmap_path = cfg.get("label_map_path", None)
        if lmap_path and os.path.exists(lmap_path):
            with open(lmap_path) as f:
                self.label_map: dict = json.load(f)
        else:
            self.label_map = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_label(self, path: str) -> int:
        if path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
            raw = d.get(self.label_col, 0)
        else:
            import h5py
            with h5py.File(path, "r") as hf:
                raw = int(hf[self.label_col][()])

        if isinstance(raw, (bytes, str)):
            raw = raw.decode() if isinstance(raw, bytes) else raw
            return self.label_map.get(raw, 0)
        return int(raw)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        label  = self._load_label(self.files[idx])
        sample["label"] = torch.tensor(label, dtype=torch.long)
        return sample
