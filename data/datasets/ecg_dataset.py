"""
data/datasets/ecg_dataset.py

Phase 3 Pre-training용 Dataset.
10초 12-lead ECG 하나를 처리해 아래를 반환:
  beats    : (N_beats, 12, beat_length)  float32
  rr_feats : (N_beats, 12, 3)            float32
  stft     : (12, F, T')                 float32

파일 구조:
  data_dir/{split}/
    record_001.npy  또는  record_001.h5

npy 포맷: dict np.save 사용 권장
  {
    "signal"  : (12, T)       원본 ECG 신호
    "rpeaks"  : (N,)          R-peak 인덱스 (선택, 없으면 자동 검출)
  }

h5 포맷:
  hf["signal"]  : (12, T)
  hf["rpeaks"]  : (N,)         (선택)
"""

from __future__ import annotations
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing.beat_segmentor import process_ecg_record
from data.preprocessing.resampler      import resample_beat, normalize_beat
from data.preprocessing.stft_extractor import compute_stft_map


class ECGDataset(Dataset):
    """
    Pre-training용 10초 ECG Dataset.
    __getitem__ 은 하나의 ECG 레코드를 처리해 dict 반환.
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg         = cfg
        self.beat_length = cfg.get("beat_length", 256)
        self.fs          = cfg.get("fs", 500)
        self.max_beats   = cfg.get("max_beats_per_lead", 15)
        self.n_leads     = cfg.get("n_leads", 12)
        self.stft_n_fft  = cfg.get("stft_n_fft", 256)
        self.stft_hop    = cfg.get("stft_hop", 64)

        data_dir = os.path.join(cfg["data_dir"], split)
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npy")) +
            glob.glob(os.path.join(data_dir, "*.h5"))
        )
        assert self.files, f"No files in {data_dir}"
        print(f"[ECGDataset:{split}] {len(self.files):,} records")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_file(self, path: str):
        if path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
            signal = d["signal"].astype(np.float32)      # (12, T)
            rpeaks = d.get("rpeaks", None)
        else:
            import h5py
            with h5py.File(path, "r") as hf:
                signal = hf["signal"][:].astype(np.float32)
                rpeaks = hf["rpeaks"][:] if "rpeaks" in hf else None
        return signal, rpeaks

    def _pad_or_trim(self, beats_arr: np.ndarray, rr_list: list) -> tuple:
        """N_beats 를 max_beats 로 맞춤 (pad or trim)."""
        N = beats_arr.shape[0]
        L, W = beats_arr.shape[1], beats_arr.shape[2]

        if N >= self.max_beats:
            return beats_arr[:self.max_beats], rr_list[:self.max_beats]

        # pad with zeros
        pad = np.zeros((self.max_beats - N, L, W), dtype=np.float32)
        beats_arr = np.concatenate([beats_arr, pad], axis=0)

        dummy_rr = {"prev_rr": 0.0, "next_rr": 0.0, "median_rr": 0.0}
        rr_list  = rr_list + [dummy_rr] * (self.max_beats - N)
        return beats_arr, rr_list

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        signal, rpeaks_hint = self._load_file(path)

        # ── Beat segmentation ────────────────────────────────────────────────
        result = process_ecg_record(
            signal, self.fs,
            rpeak_method="neurokit",
        )
        if result is None or result["n_beats"] < 2:
            # fallback: return zeros (collate_fn 에서 필터링 권장)
            return self._zero_sample()

        beats_raw = result["beats"]      # (N, 12, W_raw)
        rr_feats  = result["rr_feats"]   # list of N dicts

        # ── Resample + normalize each beat ───────────────────────────────────
        N = beats_raw.shape[0]
        beats_proc = np.zeros((N, self.n_leads, self.beat_length), dtype=np.float32)
        for b in range(N):
            for l in range(self.n_leads):
                seg = beats_raw[b, l, :]
                seg = resample_beat(seg, self.beat_length)
                seg = normalize_beat(seg[np.newaxis], "zscore")[0]
                beats_proc[b, l, :] = seg

        # ── Pad / trim to max_beats ──────────────────────────────────────────
        beats_proc, rr_feats = self._pad_or_trim(beats_proc, rr_feats)
        # beats_proc: (max_beats, 12, beat_length)

        # ── RR features -> (max_beats, 12, 3) ───────────────────────────────
        rr_arr = np.zeros((self.max_beats, self.n_leads, 3), dtype=np.float32)
        for b, rr in enumerate(rr_feats):
            rr_vec = np.array([rr["prev_rr"], rr["next_rr"], rr["median_rr"]],
                              dtype=np.float32)
            rr_arr[b, :, :] = rr_vec[np.newaxis, :]  # broadcast across leads

        # ── STFT global context ──────────────────────────────────────────────
        stft = compute_stft_map(signal, self.fs,
                                n_fft=self.stft_n_fft,
                                hop_length=self.stft_hop)  # (12, F, T')

        return {
            "beats":    torch.from_numpy(beats_proc),   # (N, 12, W)
            "rr_feats": torch.from_numpy(rr_arr),        # (N, 12, 3)
            "stft":     torch.from_numpy(stft),          # (12, F, T')
        }

    def _zero_sample(self) -> dict:
        F = self.stft_n_fft // 2 + 1
        T_stft = self.cfg.get("fs", 500) * 10 // self.stft_hop + 1
        return {
            "beats":    torch.zeros(self.max_beats, self.n_leads, self.beat_length),
            "rr_feats": torch.zeros(self.max_beats, self.n_leads, 3),
            "stft":     torch.zeros(self.n_leads, F, T_stft),
        }
