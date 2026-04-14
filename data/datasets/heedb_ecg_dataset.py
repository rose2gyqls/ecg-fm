"""
data/datasets/heedb_ecg_dataset.py

Phase 3 Pretrain용 HEEDB Dataset (ECGDataset 의 HEEDB 버전).
가변 fs 리샘플 + HEEDB 경로 직접 접근 + (옵션) 사전계산 R-peak 재사용.
"""

from __future__ import annotations
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing.heedb_io       import load_heedb_record, align_to_heedb_order
from data.preprocessing.beat_segmentor import (
    process_ecg_record, extract_beats, detect_rpeaks, compute_rr_features, LEAD_II_INDEX,
)
from data.preprocessing.resampler      import (
    resample_signal, resample_beat, normalize_beat,
)
from data.preprocessing.stft_extractor import compute_stft_map


class HEEDBECGDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg         = cfg
        self.target_fs   = int(cfg.get("target_fs", cfg.get("fs", 500)))
        self.beat_length = int(cfg.get("beat_length", 256))
        self.max_beats   = int(cfg.get("max_beats_per_lead", 15))
        self.n_leads     = int(cfg.get("n_leads", 12))
        self.stft_n_fft  = int(cfg.get("stft_n_fft", 256))
        self.stft_hop    = int(cfg.get("stft_hop", 64))
        self.before_ms   = int(cfg.get("before_ms", 200))
        self.after_ms    = int(cfg.get("after_ms", 400))

        data_dir = cfg["data_dir"]
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            self.files = sorted(glob.glob(os.path.join(split_dir, "**", "*.h5"),
                                          recursive=True))
        else:
            all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"),
                                         recursive=True))
            rng = random.Random(cfg.get("seed", 42))
            rng.shuffle(all_files)
            cut = int(len(all_files) * float(cfg.get("train_ratio", 0.95)))
            self.files = all_files[:cut] if split == "train" else all_files[cut:]

        assert self.files, f"No HEEDB files for {split}"
        print(f"[HEEDBECGDataset:{split}] {len(self.files):,} records")

    # ── Dataset ──────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.files)

    def _zero_sample(self) -> dict:
        F = self.stft_n_fft // 2 + 1
        T_stft = self.target_fs * 10 // self.stft_hop + 1
        return {
            "beats":    torch.zeros(self.max_beats, self.n_leads, self.beat_length),
            "rr_feats": torch.zeros(self.max_beats, self.n_leads, 3),
            "stft":     torch.zeros(self.n_leads, F, T_stft),
        }

    def __getitem__(self, idx: int) -> dict:
        rec = load_heedb_record(self.files[idx], load_rpeaks=True)
        if rec is None:
            return self._zero_sample()

        signal = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if signal is None:
            return self._zero_sample()

        fs_in  = rec["fs"]
        rpeaks = rec["rpeaks"]

        if fs_in != self.target_fs:
            signal = resample_signal(signal, fs_in, self.target_fs)
            if rpeaks is not None:
                rpeaks = (rpeaks.astype(np.float64)
                          * self.target_fs / fs_in).astype(np.int64)
        fs = self.target_fs

        if rpeaks is None or len(rpeaks) < 2:
            try:
                rpeaks = detect_rpeaks(signal[LEAD_II_INDEX], fs, method="neurokit")
            except Exception:
                return self._zero_sample()

        if len(rpeaks) < 2:
            return self._zero_sample()

        # 12-lead 비트
        beats_list = extract_beats(signal, rpeaks, fs,
                                   before_ms=self.before_ms,
                                   after_ms=self.after_ms)
        beats_raw = np.stack(beats_list, axis=0) if beats_list else None
        if beats_raw is None:
            return self._zero_sample()
        rr_feats = compute_rr_features(rpeaks, fs)

        # resample + normalize
        N = beats_raw.shape[0]
        beats_proc = np.zeros((N, self.n_leads, self.beat_length), dtype=np.float32)
        for b in range(N):
            for l in range(self.n_leads):
                seg = resample_beat(beats_raw[b, l, :], self.beat_length)
                beats_proc[b, l, :] = normalize_beat(seg[np.newaxis], "zscore")[0]

        # pad/trim
        if N >= self.max_beats:
            beats_proc = beats_proc[:self.max_beats]
            rr_feats   = rr_feats[:self.max_beats]
        else:
            pad = np.zeros((self.max_beats - N, self.n_leads, self.beat_length),
                           dtype=np.float32)
            beats_proc = np.concatenate([beats_proc, pad], axis=0)
            rr_feats += [{"prev_rr":0.0,"next_rr":0.0,"median_rr":0.0}] * (self.max_beats - N)

        rr_arr = np.zeros((self.max_beats, self.n_leads, 3), dtype=np.float32)
        for b, rr in enumerate(rr_feats):
            rr_arr[b, :, :] = np.array(
                [rr["prev_rr"], rr["next_rr"], rr["median_rr"]], dtype=np.float32
            )[np.newaxis, :]

        stft = compute_stft_map(signal, fs,
                                n_fft=self.stft_n_fft,
                                hop_length=self.stft_hop)

        return {
            "beats":    torch.from_numpy(beats_proc),
            "rr_feats": torch.from_numpy(rr_arr),
            "stft":     torch.from_numpy(stft),
        }
