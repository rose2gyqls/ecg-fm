from __future__ import annotations
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing.heedb_io       import load_heedb_record, align_to_heedb_order
from data.preprocessing.beat_segmentor import (
    process_ecg_record, extract_beats, detect_rpeaks, compute_rr_features,
    compute_qrs_intervals, LEAD_II_INDEX,
)
from data.preprocessing.resampler      import (
    resample_signal, resample_beat, normalize_beat,
    compute_record_norm_stats, apply_record_norm,
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
        fid_cfg = cfg.get("fiducial", {}) or {}
        self.fid_q_window_ms = int(fid_cfg.get("q_window_ms", 50))
        self.fid_s_window_ms = int(fid_cfg.get("s_window_ms", 80))
        self.normalize = cfg.get("normalize", "record_mad")
        self.record_mad_scale = float(cfg.get("record_mad_scale", 5.0))
        if self.normalize not in ("record_mad", "zscore", "none"):
            raise ValueError(f"unknown normalize mode: {self.normalize}")
        split_key = f"{split}_list"
        if cfg.get(split_key):
            with open(cfg[split_key]) as f:
                self.files = [ln.strip() for ln in f if ln.strip()]
        else:
            data_dir = cfg["data_dir"]
            split_dir = os.path.join(data_dir, split)
            if os.path.isdir(split_dir):
                self.files = sorted(glob.glob(
                    os.path.join(split_dir, "**", "*.h5"), recursive=True
                ))
            else:
                all_files = sorted(glob.glob(
                    os.path.join(data_dir, "**", "*.h5"), recursive=True
                ))
                rng = random.Random(cfg.get("seed", 42))
                rng.shuffle(all_files)
                cut = int(len(all_files) * float(cfg.get("train_ratio", 0.95)))
                self.files = all_files[:cut] if split == "train" else all_files[cut:]
        assert self.files, f"No HEEDB files for {split}"
        print(f"[HEEDBECGDataset:{split}] {len(self.files):,} records")
    def __len__(self):
        return len(self.files)
    def _zero_sample(self) -> dict:
        F = self.stft_n_fft // 2 + 1
        T_stft = self.target_fs * 10 // self.stft_hop + 1
        return {
            "beats":    torch.zeros(self.max_beats, self.n_leads, self.beat_length),
            "rr_feats": torch.zeros(self.max_beats, self.n_leads, 3),
            "fid_feats": torch.zeros(self.max_beats, self.n_leads, 2),
            "stft":     torch.zeros(self.n_leads, F, T_stft),
            "beat_t_starts":   torch.zeros(self.max_beats, dtype=torch.long),
            "beat_t_ends":     torch.zeros(self.max_beats, dtype=torch.long),
            "beat_valid_mask": torch.zeros(self.max_beats, dtype=torch.bool),
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
        beats_list = extract_beats(signal, rpeaks, fs,
                                   before_ms=self.before_ms,
                                   after_ms=self.after_ms)
        beats_raw = np.stack(beats_list, axis=0) if beats_list else None
        if beats_raw is None:
            return self._zero_sample()
        rr_feats = compute_rr_features(rpeaks, fs)
        if self.normalize == "record_mad":
            rec_med, rec_mad = compute_record_norm_stats(signal)
        N = beats_raw.shape[0]
        beats_proc = np.zeros((N, self.n_leads, self.beat_length), dtype=np.float32)
        for b in range(N):
            for l in range(self.n_leads):
                seg = resample_beat(beats_raw[b, l, :], self.beat_length)
                if self.normalize == "zscore":
                    seg = normalize_beat(seg[np.newaxis], "zscore")[0]
                elif self.normalize == "record_mad":
                    seg = apply_record_norm(seg, rec_med, rec_mad,
                                            scale=self.record_mad_scale)
                beats_proc[b, l, :] = seg
        if N >= self.max_beats:
            beats_proc = beats_proc[:self.max_beats]
            rr_feats   = rr_feats[:self.max_beats]
            n_valid = self.max_beats
        else:
            pad = np.zeros((self.max_beats - N, self.n_leads, self.beat_length),
                           dtype=np.float32)
            beats_proc = np.concatenate([beats_proc, pad], axis=0)
            rr_feats += [{"prev_rr":0.0,"next_rr":0.0,"median_rr":0.0}] * (self.max_beats - N)
            n_valid = N
        rr_arr = np.zeros((self.max_beats, self.n_leads, 3), dtype=np.float32)
        for b, rr in enumerate(rr_feats):
            rr_arr[b, :, :] = np.array(
                [rr["prev_rr"], rr["next_rr"], rr["median_rr"]], dtype=np.float32
            )[np.newaxis, :]
        fid_arr = compute_qrs_intervals(
            beats_proc,
            before_ms=self.before_ms,
            after_ms=self.after_ms,
            beat_length=self.beat_length,
            q_window_ms=self.fid_q_window_ms,
            s_window_ms=self.fid_s_window_ms,
        )
        if n_valid < self.max_beats:
            fid_arr[n_valid:] = 0.0
        stft = compute_stft_map(signal, fs,
                                n_fft=self.stft_n_fft,
                                hop_length=self.stft_hop)
        before_samples = int(round(fs * self.before_ms / 1000))
        after_samples  = int(round(fs * self.after_ms  / 1000))
        T_stft = stft.shape[-1]
        half_fft = self.stft_n_fft // 2
        beat_t_starts = np.zeros(self.max_beats, dtype=np.int64)
        beat_t_ends   = np.zeros(self.max_beats, dtype=np.int64)
        valid_rpeaks = np.asarray(rpeaks[:n_valid], dtype=np.int64)
        if n_valid > 0:
            ts = np.maximum(0, (valid_rpeaks - before_samples - half_fft) // self.stft_hop)
            te = np.minimum(T_stft,
                            (valid_rpeaks + after_samples + half_fft) // self.stft_hop + 1)
            te = np.maximum(te, ts + 1)
            te = np.minimum(te, T_stft)
            beat_t_starts[:n_valid] = ts
            beat_t_ends[:n_valid]   = te
        beat_valid_mask = np.zeros(self.max_beats, dtype=bool)
        beat_valid_mask[:n_valid] = True
        return {
            "beats":    torch.from_numpy(beats_proc),
            "rr_feats": torch.from_numpy(rr_arr),
            "fid_feats": torch.from_numpy(fid_arr),
            "stft":     torch.from_numpy(stft),
            "beat_t_starts":   torch.from_numpy(beat_t_starts),
            "beat_t_ends":     torch.from_numpy(beat_t_ends),
            "beat_valid_mask": torch.from_numpy(beat_valid_mask),
        }
