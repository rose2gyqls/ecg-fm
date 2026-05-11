from __future__ import annotations
import os
import glob
import json
import numpy as np
import torch
from data.datasets.ecg_dataset import ECGDataset

class FinetuneDataset(ECGDataset):
    def __init__(self, cfg: dict, split: str = "train"):
        super().__init__(cfg, split)
        self.label_col = cfg.get("label_col", "label")
        lmap_path = cfg.get("label_map_path", None)
        if lmap_path and os.path.exists(lmap_path):
            with open(lmap_path) as f:
                self.label_map: dict = json.load(f)
        else:
            self.label_map = {}
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
    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        label  = self._load_label(self.files[idx])
        sample["label"] = torch.tensor(label, dtype=torch.long)
        return sample
