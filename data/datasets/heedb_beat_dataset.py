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
)
from data.preprocessing.resampler      import (
    resample_signal, resample_beat, normalize_beat,
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
        normalize       : "zscore"
        max_beats_per_record : int, 레코드당 최대 사용 비트 (랜덤 샘플)
        cache           : True이면 첫 epoch에 메모리로 비트 캐시
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.target_fs   = int(cfg.get("target_fs", cfg.get("fs", 500)))
        self.beat_length = int(cfg.get("beat_length", 256))
        self.before_ms   = int(cfg.get("before_ms", 200))
        self.after_ms    = int(cfg.get("after_ms", 400))
        self.normalize   = cfg.get("normalize", "zscore")
        self.max_per_rec = int(cfg.get("max_beats_per_record", 10))
        self.cache_mode  = bool(cfg.get("cache", False))

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
        rec = load_heedb_record(path, load_rpeaks=True)
        if rec is None:
            return None

        signal = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if signal is None:
            return None

        fs_in = rec["fs"]
        # 원본 R-peak 우선 사용 → 없거나 불충분하면 Lead II에서 재검출
        rpeaks_raw = rec["rpeaks"]
        # fs 리샘플 (비트 자르기 전에 공통 fs로 맞춤)
        if fs_in != self.target_fs:
            signal = resample_signal(signal, fs_in, self.target_fs)
            if rpeaks_raw is not None:
                rpeaks_raw = (rpeaks_raw.astype(np.float64)
                              * self.target_fs / fs_in).astype(np.int64)

        fs = self.target_fs
        if rpeaks_raw is None or len(rpeaks_raw) < 2:
            try:
                rpeaks = detect_rpeaks(signal[LEAD_II_INDEX], fs, method="neurokit")
            except Exception:
                return None
        else:
            rpeaks = rpeaks_raw

        if len(rpeaks) < 2:
            return None

        # 12-lead 비트 추출 (동일 R-peak → 리드별 동일 개수)
        beats_arr = extract_beats(signal, rpeaks, fs,
                                  before_ms=self.before_ms,
                                  after_ms=self.after_ms)
        # extract_beats: 다채널 입력 시 list of (12, W) 반환
        if len(beats_arr) == 0:
            return None
        beats = np.stack(beats_arr, axis=0)              # (N, 12, W_raw)

        # 필요 시 레코드당 비트 수 제한 (랜덤)
        if self.max_per_rec and beats.shape[0] > self.max_per_rec:
            sel = np.random.choice(beats.shape[0], self.max_per_rec, replace=False)
            beats = beats[np.sort(sel)]

        # beat_length 로 리샘플 + lead flatten
        N, L, W = beats.shape
        beats = beats.reshape(N * L, W)                  # (N*12, W_raw)
        beats = resample_beat(beats, self.beat_length)   # (N*12, beat_length)
        return beats

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self):
        if self._cache is not None:
            return len(self._cache)
        # streaming 모드: record × max_per_rec × 12 추정치
        return len(self.files) * self.max_per_rec * 12

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            beat = self._cache[idx].copy()
        else:
            # streaming: 유효 비트가 나올 때까지 레코드를 탐색
            for _ in range(8):
                f = self.files[random.randrange(len(self.files))]
                arr = self._extract_record_beats(f)
                if arr is None or len(arr) == 0:
                    continue
                beat = arr[random.randrange(len(arr))].copy()
                break
            else:
                beat = np.zeros(self.beat_length, dtype=np.float32)

        beat = normalize_beat(beat[np.newaxis], self.normalize)  # (1, W)
        return {"beat": torch.from_numpy(beat.astype(np.float32))}


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
