"""
data/datasets/heedb_beat_dataset.py

HEEDB H5 원본을 직접 읽어 VQ-VAE(Phase 1) 학습용 비트를 반환.
- Lead II 기반 R-peak (beat_annotation 있으면 재사용, 없으면 neurokit2)
- 가변 fs -> target_fs 리샘플
- 12개 리드 모두를 같은 R-peak 위치로 잘라 (N, 12, W) 생성
- __getitem__ 은 하나의 (단일 리드, 단일 비트)를 반환 → 코드북이 리드 전반을 학습
"""

from __future__ import annotations
import os
import glob
import random
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
    """
    VQ-VAE 학습용. 매 epoch마다 레코드를 순회하며 (12, N_beats) 비트를 flatten.

    cfg 키:
        data_dir        : HEEDB *.h5 루트
        file_list       : (선택) 파일 경로 리스트 txt
        target_fs       : 500
        beat_length     : 256
        before_ms       : 200
        after_ms        : 400
        normalize       : "record_mad" | "zscore" | "none"
                          - record_mad: per-record (median, MAD)·5 scaling.
                            Preserves inter-lead amplitude (V1 vs V6).
                          - zscore   : per-beat per-lead z-score (legacy).
                          - none     : no normalization.
        max_beats_per_record : int, 레코드당 최대 사용 비트 (랜덤 샘플)
        cache           : True이면 첫 epoch에 메모리로 비트 캐시
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.target_fs   = int(cfg.get("target_fs", cfg.get("fs", 500)))
        self.beat_length = int(cfg.get("beat_length", 256))
        self.before_ms   = int(cfg.get("before_ms", 200))
        self.after_ms    = int(cfg.get("after_ms", 400))
        self.normalize   = cfg.get("normalize", "record_mad")
        self.record_mad_scale = float(cfg.get("record_mad_scale", 5.0))
        if self.normalize not in ("record_mad", "zscore", "none"):
            raise ValueError(f"unknown normalize mode: {self.normalize}")
        self.max_per_rec = int(cfg.get("max_beats_per_record", 10))
        self.cache_mode  = bool(cfg.get("cache", False))

        # R-peak validation & noise filter (configurable, sensible defaults)
        rv = cfg.get("rpeak_validation", {}) or {}
        self.rv_local_max         = bool(rv.get("local_max", True))
        self.rv_local_max_window  = int(rv.get("local_max_window", 10))
        self.rv_local_max_shift   = int(rv.get("local_max_shift", 8))

        nf = cfg.get("noise_filter", {}) or {}
        self.nf_enabled = bool(nf.get("enabled", True))
        self.nf_ptp_min = float(nf.get("ptp_min", 0.1))    # mV
        self.nf_std_min = float(nf.get("std_min", 0.01))   # mV

        # Virtual epoch size (streaming mode). split별 독립 지정.
        vlen_cfg = cfg.get("virtual_len", {}) or {}
        self._virtual_len: Optional[int] = (
            int(vlen_cfg[split]) if split in vlen_cfg else None
        )

        # 워커별 beat buffer — record 하나를 처리하면 얻는 ~120 beats를
        # 버리지 않고 순차적으로 공급. __getitem__당 record 1개 처리 대신
        # ~120번 호출당 1개 처리로 throughput ~120× 향상.
        self._buf: Optional[np.ndarray] = None
        self._buf_idx: int = 0

        self.files = _resolve_files(cfg, split)
        assert self.files, f"No HEEDB files for split={split}"
        print(f"[HEEDBBeatDataset:{split}] {len(self.files):,} records")

        self._cache: Optional[np.ndarray] = None
        if self.cache_mode:
            self._build_cache()

    # ── cache 모드: 모든 비트를 (M, W) 로 미리 적재 ────────────────────────────
    def _build_cache(self):
        chunks = []
        for p in self.files:
            arr = self._extract_record_beats(p)          # (N*12, W) or None
            if arr is not None and len(arr) > 0:
                chunks.append(arr)
        if not chunks:
            raise RuntimeError("No valid beats extracted from HEEDB.")
        self._cache = np.concatenate(chunks, axis=0).astype(np.float32)
        print(f"[HEEDBBeatDataset] cache built: {len(self._cache):,} beats")

    # ── 단일 레코드 → (M, W) 비트 배열 ─────────────────────────────────────────
    def _extract_record_beats(self, path: str) -> Optional[np.ndarray]:
        rec = load_heedb_record(path, load_rpeaks=False)
        if rec is None:
            return None

        signal = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if signal is None:
            return None

        fs_in = rec["fs"]
        # 일관성을 위해 H5 내장 annotation은 무시하고 항상 Lead II + neurokit으로 재검출.
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

        # 1) 경계 거리 필터: before/after 윈도우가 완전히 신호 안에 들어오도록
        before_samp = int(fs * self.before_ms / 1000)
        after_samp  = int(fs * self.after_ms  / 1000)
        rpeaks = validate_rpeaks_boundary(rpeaks, ref.shape[-1],
                                          before_samp, after_samp)

        # 2) 국소 극대점 검증 (Lead II 기준)
        if self.rv_local_max and len(rpeaks) > 0:
            rpeaks = validate_rpeaks_local_max(
                ref, rpeaks,
                window=self.rv_local_max_window,
                max_shift=self.rv_local_max_shift,
            )

        if len(rpeaks) < 2:
            return None

        # 12-lead 비트 추출 (동일 R-peak → 리드별 동일 개수, padding 없이 전부 in-bounds)
        beats_arr = extract_beats(signal, rpeaks, fs,
                                  before_ms=self.before_ms,
                                  after_ms=self.after_ms)
        if len(beats_arr) == 0:
            return None
        beats = np.stack(beats_arr, axis=0)              # (N, 12, W_raw)

        # 3) noise/flat beat 필터 — (beat, lead) 단위로 drop. raw mV thresholds.
        N, L, W = beats.shape
        beats = beats.reshape(N * L, W)                  # (N*12, W_raw)
        if self.nf_enabled:
            flat = flat_beat_mask(beats,
                                  ptp_min=self.nf_ptp_min,
                                  std_min=self.nf_std_min)
            beats = beats[~flat]
        if beats.shape[0] == 0:
            return None

        # 4) record-level normalization (mode=="record_mad"): apply ONCE using
        #    record-level (median, MAD) computed on the full (12, T) raw signal.
        #    Per-beat z-score happens later in __getitem__ for mode=="zscore".
        if self.normalize == "record_mad":
            med, mad = compute_record_norm_stats(signal)
            beats = apply_record_norm(beats, med, mad, scale=self.record_mad_scale)

        # 레코드당 비트 수 제한 (랜덤, lead-flattened 이후에 적용)
        limit = self.max_per_rec * L if self.max_per_rec else beats.shape[0]
        if beats.shape[0] > limit:
            sel = np.random.choice(beats.shape[0], limit, replace=False)
            beats = beats[np.sort(sel)]

        # beat_length 로 리샘플
        beats = resample_beat(beats, self.beat_length)   # (M, beat_length)
        return beats

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self):
        if self._cache is not None:
            return len(self._cache)
        if self._virtual_len is not None:
            return self._virtual_len
        # streaming 모드 기본값: record × max_per_rec × 12 추정치
        return len(self.files) * self.max_per_rec * 12

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            beat = self._cache[idx].copy()
        else:
            # streaming: 워커의 beat buffer에서 순차 공급. 비면 다음 record에서 리필.
            if self._buf is None or self._buf_idx >= len(self._buf):
                self._refill_buffer()

            if self._buf is None or len(self._buf) == 0:
                beat = np.zeros(self.beat_length, dtype=np.float32)
            else:
                beat = self._buf[self._buf_idx].copy()
                self._buf_idx += 1

        # If mode=="record_mad", normalization was already applied in
        # _extract_record_beats. Only z-score needs per-beat application here.
        if self.normalize == "zscore":
            beat = normalize_beat(beat[np.newaxis], "zscore")  # (1, W)
        else:
            beat = beat[np.newaxis]
        return {"beat": torch.from_numpy(beat.astype(np.float32))}

    def _refill_buffer(self):
        """유효 beats가 나올 때까지 최대 8 record 시도. 실패 시 빈 버퍼 유지."""
        for _ in range(8):
            f = self.files[random.randrange(len(self.files))]
            arr = self._extract_record_beats(f)
            if arr is not None and len(arr) > 0:
                np.random.shuffle(arr)   # 동일 record 연속 공급 시 lead 혼합
                self._buf = arr
                self._buf_idx = 0
                return
        self._buf = None
        self._buf_idx = 0


# ── 파일 리스트 해결 ───────────────────────────────────────────────────────────
def _resolve_files(cfg: dict, split: str) -> List[str]:
    # 1) split별 파일 리스트 (train_list / val_list) — 가장 메모리 효율적
    split_key = f"{split}_list"
    if cfg.get(split_key):
        with open(cfg[split_key]) as f:
            return [ln.strip() for ln in f if ln.strip()]

    # 2) 단일 file_list를 train_ratio로 분할
    if cfg.get("file_list"):
        with open(cfg["file_list"]) as f:
            files = [ln.strip() for ln in f if ln.strip()]
        train_ratio = float(cfg.get("train_ratio", 0.9))
        rng = random.Random(cfg.get("seed", 42))
        rng.shuffle(files)
        cut = int(len(files) * train_ratio)
        return files[:cut] if split == "train" else files[cut:]

    # 3) data_dir/{split}/*.h5
    data_dir = cfg["data_dir"]
    split_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "**", "*.h5"), recursive=True))
    if files:
        return files

    # 4) data_dir/**/*.h5  를 train/val 비율로 나누기
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    if not files:
        return []
    train_ratio = float(cfg.get("train_ratio", 0.9))
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(files)
    cut = int(len(files) * train_ratio)
    return files[:cut] if split == "train" else files[cut:]
