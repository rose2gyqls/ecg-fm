from __future__ import annotations
import os
import glob
import random
from collections import OrderedDict
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing.heedb_io       import load_heedb_record, align_to_heedb_order
from data.preprocessing.beat_segmentor import (
    detect_rpeaks, extract_beats, LEAD_II_INDEX,
    validate_rpeaks_boundary, validate_rpeaks_local_max,
    flat_beat_mask,
)
from data.preprocessing.resampler      import (
    resample_signal, resample_beat, normalize_beat,
    compute_record_norm_stats, apply_record_norm,
)

class HEEDBBeatDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.target_fs   = int(cfg.get("target_fs", cfg.get("fs", 500)))
        self.beat_length = int(cfg.get("beat_length", 256))
        self.before_ms   = int(cfg.get("before_ms", 200))
        self.after_ms    = int(cfg.get("after_ms", 400))
        self.normalize   = cfg.get("normalize", "record_mad")
        self.record_mad_scale = float(cfg.get("record_mad_scale", 5.0))
        self.record_mad_min_scale = float(cfg.get("record_mad_min_scale", 0.05))
        rm_clip = cfg.get("record_mad_clip", None)
        self.record_mad_clip: Optional[float] = (
            None if rm_clip is None else float(rm_clip)
        )
        if self.normalize not in ("record_mad", "zscore", "none"):
            raise ValueError(f"unknown normalize mode: {self.normalize}")
        self.max_per_rec = int(cfg.get("max_beats_per_record", 10))
        self.cache_mode  = bool(cfg.get("cache", False))
        rv = cfg.get("rpeak_validation", {}) or {}
        self.rv_local_max         = bool(rv.get("local_max", True))
        self.rv_local_max_window  = int(rv.get("local_max_window", 10))
        self.rv_local_max_shift   = int(rv.get("local_max_shift", 8))
        nf = cfg.get("noise_filter", {}) or {}
        self.nf_enabled = bool(nf.get("enabled", True))
        self.nf_ptp_min = float(nf.get("ptp_min", 0.1))
        self.nf_std_min = float(nf.get("std_min", 0.01))
        vlen_cfg = cfg.get("virtual_len", {}) or {}
        self._virtual_len: Optional[int] = (
            int(vlen_cfg[split]) if split in vlen_cfg else None
        )
        self._buf: Optional[np.ndarray] = None
        self._buf_idx: int = 0
        self._record_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_cap = int(cfg.get("record_cache_size", 2000))
        self._cache_hits = 0
        self._cache_misses = 0
        self.files = _resolve_files(cfg, split)
        assert self.files, f"No HEEDB files for split={split}"
        print(f"[HEEDBBeatDataset:{split}] {len(self.files):,} records  "
              f"record_cache_cap={self._cache_cap}")
        self._cache: Optional[np.ndarray] = None
        if self.cache_mode:
            self._build_cache()
    def _build_cache(self):
        chunks = []
        for p in self.files:
            arr = self._extract_record_beats(p)
            if arr is not None and len(arr) > 0:
                chunks.append(arr)
        if not chunks:
            raise RuntimeError("No valid beats extracted from HEEDB.")
        self._cache = np.concatenate(chunks, axis=0).astype(np.float32)
        print(f"[HEEDBBeatDataset] cache built: {len(self._cache):,} beats")
    def _extract_record_beats(self, path: str) -> Optional[np.ndarray]:
        rec = load_heedb_record(path, load_rpeaks=False)
        if rec is None:
            return None
        signal = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if signal is None:
            return None
        fs_in = rec["fs"]
        if fs_in != self.target_fs:
            signal = resample_signal(signal, fs_in, self.target_fs)
        fs = self.target_fs
        ref = signal[LEAD_II_INDEX]
        try:
            rpeaks = detect_rpeaks(ref, fs, method="neurokit")
        except Exception:
            return None
        if len(rpeaks) < 2:
            return None
        before_samp = int(fs * self.before_ms / 1000)
        after_samp  = int(fs * self.after_ms  / 1000)
        rpeaks = validate_rpeaks_boundary(rpeaks, ref.shape[-1],
                                          before_samp, after_samp)
        if self.rv_local_max and len(rpeaks) > 0:
            rpeaks = validate_rpeaks_local_max(
                ref, rpeaks,
                window=self.rv_local_max_window,
                max_shift=self.rv_local_max_shift,
            )
        if len(rpeaks) < 2:
            return None
        beats_arr = extract_beats(signal, rpeaks, fs,
                                  before_ms=self.before_ms,
                                  after_ms=self.after_ms)
        if len(beats_arr) == 0:
            return None
        beats = np.stack(beats_arr, axis=0)
        N, L, W = beats.shape
        beats = beats.reshape(N * L, W)
        if self.nf_enabled:
            flat = flat_beat_mask(beats,
                                  ptp_min=self.nf_ptp_min,
                                  std_min=self.nf_std_min)
            beats = beats[~flat]
        if beats.shape[0] == 0:
            return None
        if self.normalize == "record_mad":
            med, mad = compute_record_norm_stats(
                signal, min_scale=self.record_mad_min_scale,
            )
            beats = apply_record_norm(
                beats, med, mad,
                scale=self.record_mad_scale,
                clip=self.record_mad_clip,
            )
        limit = self.max_per_rec * L if self.max_per_rec else beats.shape[0]
        if beats.shape[0] > limit:
            sel = np.random.choice(beats.shape[0], limit, replace=False)
            beats = beats[np.sort(sel)]
        beats = resample_beat(beats, self.beat_length)
        return beats
    def __len__(self):
        if self._cache is not None:
            return len(self._cache)
        if self._virtual_len is not None:
            return self._virtual_len
        return len(self.files) * self.max_per_rec * 12
    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            beat = self._cache[idx].copy()
        else:
            if self._buf is None or self._buf_idx >= len(self._buf):
                self._refill_buffer()
            if self._buf is None or len(self._buf) == 0:
                beat = np.zeros(self.beat_length, dtype=np.float32)
            else:
                beat = self._buf[self._buf_idx].copy()
                self._buf_idx += 1
        if self.normalize == "zscore":
            beat = normalize_beat(beat[np.newaxis], "zscore")
        else:
            beat = beat[np.newaxis]
        return {"beat": torch.from_numpy(beat.astype(np.float32))}
    def _refill_buffer(self):
        for _ in range(8):
            f = self.files[random.randrange(len(self.files))]
            arr = self._record_cache.get(f)
            if arr is not None:
                self._record_cache.move_to_end(f)
                self._cache_hits += 1
            else:
                arr = self._extract_record_beats(f)
                self._cache_misses += 1
                if arr is not None and len(arr) > 0:
                    self._record_cache[f] = arr
                    while len(self._record_cache) > self._cache_cap:
                        self._record_cache.popitem(last=False)
            if arr is not None and len(arr) > 0:
                self._buf = arr[np.random.permutation(len(arr))]
                self._buf_idx = 0
                return
        self._buf = None
        self._buf_idx = 0

def _resolve_files(cfg: dict, split: str) -> List[str]:
    split_key = f"{split}_list"
    if cfg.get(split_key):
        with open(cfg[split_key]) as f:
            return [ln.strip() for ln in f if ln.strip()]
    if cfg.get("file_list"):
        with open(cfg["file_list"]) as f:
            files = [ln.strip() for ln in f if ln.strip()]
        train_ratio = float(cfg.get("train_ratio", 0.9))
        rng = random.Random(cfg.get("seed", 42))
        rng.shuffle(files)
        cut = int(len(files) * train_ratio)
        return files[:cut] if split == "train" else files[cut:]
    data_dir = cfg["data_dir"]
    split_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "**", "*.h5"), recursive=True))
    if files:
        return files
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    if not files:
        return []
    train_ratio = float(cfg.get("train_ratio", 0.9))
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(files)
    cut = int(len(files) * train_ratio)
    return files[:cut] if split == "train" else files[cut:]
