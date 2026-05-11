from __future__ import annotations
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing.resampler import resample_beat, normalize_beat

class BeatDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.beat_length = cfg.get("beat_length", 256)
        self.normalize   = cfg.get("normalize", "zscore")
        if self.normalize == "record_mad":
            raise ValueError(
                "BeatDataset (file-based) cannot apply record_mad - record "
                "metadata is lost once beats are flattened into .npy/.h5. "
                "Use HEEDBBeatDataset (streaming from raw h5) instead."
            )
        data_dir = os.path.join(cfg["data_dir"], split)
        files    = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not files:
            files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            self._mode = "h5"
        else:
            self._mode = "npy"
        assert files, f"No data files found in {data_dir}"
        self._load_all(files)
    def _load_all(self, files: list[str]):
        chunks = []
        for f in files:
            if self._mode == "npy":
                arr = np.load(f, mmap_mode="r")
            else:
                import h5py
                with h5py.File(f, "r") as hf:
                    arr = hf["beats"][:]
            if arr.ndim == 3:
                N, L, W = arr.shape
                arr = arr.reshape(N * L, W)
            arr = resample_beat(arr, self.beat_length)
            chunks.append(arr)
        self.data = np.concatenate(chunks, axis=0).astype(np.float32)
        print(f"[BeatDataset] Loaded {len(self.data):,} beats.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int) -> dict:
        beat = self.data[idx].copy()
        beat = normalize_beat(beat[np.newaxis], self.normalize)
        return {"beat": torch.from_numpy(beat)}
