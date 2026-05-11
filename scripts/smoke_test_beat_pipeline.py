"""Smoke test for R-peak validation, beat extraction, and noise filtering."""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random

from data.preprocessing.heedb_io       import load_heedb_record, align_to_heedb_order
from data.preprocessing.beat_segmentor import (
    detect_rpeaks, extract_beats, flat_beat_mask,
    validate_rpeaks_boundary, validate_rpeaks_local_max,
    LEAD_II_INDEX,
)
from data.preprocessing.resampler      import resample_signal
from data.datasets.heedb_beat_dataset  import HEEDBBeatDataset


FS          = 500
BEFORE_MS   = 200
AFTER_MS    = 400
FILELIST    = os.environ.get("HEEDB_FILELIST", "file_lists/train_files_full.txt")
N_RECORDS   = 30


def ms_to_samp(ms): return int(FS * ms / 1000)


def run_record_level(paths):
    """레코드 단위로 각 단계의 drop 통계를 찍는다."""
    before = ms_to_samp(BEFORE_MS)
    after  = ms_to_samp(AFTER_MS)
    stats = {
        "records_total": 0,
        "records_invalid_load": 0,
        "records_too_few_rpeaks": 0,
        "rpeaks_raw": 0,
        "rpeaks_after_boundary": 0,
        "rpeaks_after_localmax": 0,
        "beats_raw": 0,
        "beats_after_noise": 0,
    }
    for p in paths:
        stats["records_total"] += 1
        rec = load_heedb_record(p, load_rpeaks=False)
        if rec is None:
            stats["records_invalid_load"] += 1
            continue
        sig = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if sig is None:
            stats["records_invalid_load"] += 1
            continue
        if rec["fs"] != FS:
            sig = resample_signal(sig, rec["fs"], FS)
        ref = sig[LEAD_II_INDEX]
        try:
            rp = detect_rpeaks(ref, FS, method="neurokit")
        except Exception:
            stats["records_too_few_rpeaks"] += 1
            continue
        stats["rpeaks_raw"] += len(rp)
        rp = validate_rpeaks_boundary(rp, ref.shape[-1], before, after)
        stats["rpeaks_after_boundary"] += len(rp)
        rp = validate_rpeaks_local_max(ref, rp, window=10, max_shift=8)
        stats["rpeaks_after_localmax"] += len(rp)
        if len(rp) < 2:
            stats["records_too_few_rpeaks"] += 1
            continue
        beats = extract_beats(sig, rp, FS,
                              before_ms=BEFORE_MS, after_ms=AFTER_MS)
        beats = np.stack(beats, axis=0).reshape(-1, ms_to_samp(BEFORE_MS) + ms_to_samp(AFTER_MS))
        stats["beats_raw"] += beats.shape[0]
        flat = flat_beat_mask(beats, ptp_min=0.1, std_min=0.01)
        kept = beats[~flat]
        stats["beats_after_noise"] += kept.shape[0]
    return stats


def run_dataset_level(filelist_path, n_draws=200):
    """HEEDBBeatDataset을 경유해 __getitem__이 정상 출력하는지 확인."""
    cfg = {
        "source": "heedb",
        "train_list": filelist_path,
        "val_list":   filelist_path,
        "target_fs": FS,
        "beat_length": 256,
        "before_ms": BEFORE_MS,
        "after_ms":  AFTER_MS,
        "normalize": "zscore",
        "max_beats_per_record": 10,
        "cache": False,
        "n_leads": 12,
        "seed": 42,
        "rpeak_validation": {"local_max": True, "local_max_window": 10, "local_max_shift": 8},
        "noise_filter":     {"enabled": True, "ptp_min": 0.1, "std_min": 0.01},
    }
    ds = HEEDBBeatDataset(cfg, split="train")
    shapes = set()
    zeros  = 0
    for _ in range(n_draws):
        item = ds[random.randrange(len(ds))]
        b = item["beat"].numpy()
        shapes.add(b.shape)
        if np.allclose(b, 0.0):
            zeros += 1
    return {"n_draws": n_draws, "shapes": shapes, "all_zero_draws": zeros}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", default=FILELIST)
    parser.add_argument("--n-records", type=int, default=N_RECORDS)
    args = parser.parse_args()

    random.seed(0)
    with open(args.filelist) as f:
        all_files = [ln.strip() for ln in f if ln.strip()]
    sample = random.sample(all_files, args.n_records)

    print(f"[record-level] {args.n_records} records")
    s = run_record_level(sample)
    for k, v in s.items():
        print(f"  {k:30s} {v}")

    print("\n[dataset-level] sampling via __getitem__")
    d = run_dataset_level(args.filelist, n_draws=200)
    for k, v in d.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
