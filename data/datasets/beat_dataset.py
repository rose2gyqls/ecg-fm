"""
data/datasets/beat_dataset.py

Phase 1 VQ-VAE 학습용 Dataset.
전처리된 beat segment를 (1, 256) 텐서로 반환.

지원 포맷:
  - .npy  : (N_beats, 12, 256) 또는 (N_beats, 256) 사전 저장 배열
  - .h5   : dataset key 'beats' (N, 12, 256)
  - .wfdb : 레코드 목록 + 실시간 beat 추출 (느림, 소규모 실험용)

data_dir 하위 구조 예시:
  data_dir/
    train/
      record_001.npy
      record_002.npy
      ...
    val/
      ...
"""

from __future__ import annotations
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing.resampler import resample_beat, normalize_beat


class BeatDataset(Dataset):
    """
    VQ-VAE 학습용 Dataset.
    각 __getitem__은 단일 리드의 단일 beat segment (1, W) 를 반환.

    Returns dict:
        beat : (1, beat_length) float32
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.beat_length = cfg.get("beat_length", 256)
        self.normalize   = cfg.get("normalize", "zscore")

        data_dir = os.path.join(cfg["data_dir"], split)
        files    = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        if not files:
            # h5 fallback
            files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            self._mode = "h5"
        else:
            self._mode = "npy"

        assert files, f"No data files found in {data_dir}"
        self._load_all(files)

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_all(self, files: list[str]):
        """모든 파일을 메모리에 로드 후 concat. (B, W) float32"""
        chunks = []
        for f in files:
            if self._mode == "npy":
                arr = np.load(f, mmap_mode="r")          # (N, 12, W) or (N, W)
            else:
                import h5py
                with h5py.File(f, "r") as hf:
                    arr = hf["beats"][:]                  # (N, 12, W)

            if arr.ndim == 3:
                # (N, 12, W) -> (N*12, W): 모든 리드를 flat하게
                N, L, W = arr.shape
                arr = arr.reshape(N * L, W)
            # arr: (M, W)
            arr = resample_beat(arr, self.beat_length)    # (M, 256)
            chunks.append(arr)

        self.data = np.concatenate(chunks, axis=0).astype(np.float32)
        print(f"[BeatDataset] Loaded {len(self.data):,} beats.")

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        beat = self.data[idx].copy()                      # (W,)
        beat = normalize_beat(beat[np.newaxis], self.normalize)  # (1, W)
        return {"beat": torch.from_numpy(beat)}
