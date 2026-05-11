from __future__ import annotations
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing.beat_segmentor import process_ecg_record
from data.preprocessing.resampler      import (
    resample_beat, normalize_beat,
    compute_record_norm_stats, apply_record_norm,
)
from data.preprocessing.stft_extractor import compute_stft_map

class ECGDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg         = cfg
        self.beat_length = cfg.get("beat_length", 256)
        self.fs          = cfg.get("fs", 500)
        self.max_beats   = cfg.get("max_beats_per_lead", 15)
        self.n_leads     = cfg.get("n_leads", 12)
        self.stft_n_fft  = cfg.get("stft_n_fft", 256)
        self.stft_hop    = cfg.get("stft_hop", 64)
        self.normalize = cfg.get("normalize", "record_mad")
        self.record_mad_scale = float(cfg.get("record_mad_scale", 5.0))
        if self.normalize not in ("record_mad", "zscore", "none"):
            raise ValueError(f"unknown normalize mode: {self.normalize}")
        data_dir = os.path.join(cfg["data_dir"], split)
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npy")) +
            glob.glob(os.path.join(data_dir, "*.h5"))
        )
        assert self.files, f"No files in {data_dir}"
        print(f"[ECGDataset:{split}] {len(self.files):,} records")
    def _load_file(self, path: str):
        if path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
            signal = d["signal"].astype(np.float32)
            rpeaks = d.get("rpeaks", None)
        else:
            import h5py
            with h5py.File(path, "r") as hf:
                signal = hf["signal"][:].astype(np.float32)
                rpeaks = hf["rpeaks"][:] if "rpeaks" in hf else None
        return signal, rpeaks
    def _pad_or_trim(self, beats_arr: np.ndarray, rr_list: list) -> tuple:
        N = beats_arr.shape[0]
        L, W = beats_arr.shape[1], beats_arr.shape[2]
        if N >= self.max_beats:
            return beats_arr[:self.max_beats], rr_list[:self.max_beats]
        pad = np.zeros((self.max_beats - N, L, W), dtype=np.float32)
        beats_arr = np.concatenate([beats_arr, pad], axis=0)
        dummy_rr = {"prev_rr": 0.0, "next_rr": 0.0, "median_rr": 0.0}
        rr_list  = rr_list + [dummy_rr] * (self.max_beats - N)
        return beats_arr, rr_list
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        signal, rpeaks_hint = self._load_file(path)
        result = process_ecg_record(
            signal, self.fs,
            rpeak_method="neurokit",
        )
        if result is None or result["n_beats"] < 2:
            return self._zero_sample()
        beats_raw = result["beats"]
        rr_feats  = result["rr_feats"]
        if self.normalize == "record_mad":
            rec_med, rec_mad = compute_record_norm_stats(signal)
        N = beats_raw.shape[0]
        beats_proc = np.zeros((N, self.n_leads, self.beat_length), dtype=np.float32)
        for b in range(N):
            for l in range(self.n_leads):
                seg = beats_raw[b, l, :]
                seg = resample_beat(seg, self.beat_length)
                if self.normalize == "zscore":
                    seg = normalize_beat(seg[np.newaxis], "zscore")[0]
                elif self.normalize == "record_mad":
                    seg = apply_record_norm(seg, rec_med, rec_mad,
                                            scale=self.record_mad_scale)
                beats_proc[b, l, :] = seg
        beats_proc, rr_feats = self._pad_or_trim(beats_proc, rr_feats)
        rr_arr = np.zeros((self.max_beats, self.n_leads, 3), dtype=np.float32)
        for b, rr in enumerate(rr_feats):
            rr_vec = np.array([rr["prev_rr"], rr["next_rr"], rr["median_rr"]],
                              dtype=np.float32)
            rr_arr[b, :, :] = rr_vec[np.newaxis, :]
        stft = compute_stft_map(signal, self.fs,
                                n_fft=self.stft_n_fft,
                                hop_length=self.stft_hop)
        return {
            "beats":    torch.from_numpy(beats_proc),
            "rr_feats": torch.from_numpy(rr_arr),
            "stft":     torch.from_numpy(stft),
        }
    def _zero_sample(self) -> dict:
        F = self.stft_n_fft // 2 + 1
        T_stft = self.cfg.get("fs", 500) * 10 // self.stft_hop + 1
        return {
            "beats":    torch.zeros(self.max_beats, self.n_leads, self.beat_length),
            "rr_feats": torch.zeros(self.max_beats, self.n_leads, 3),
            "stft":     torch.zeros(self.n_leads, F, T_stft),
        }
