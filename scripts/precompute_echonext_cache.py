#!/usr/bin/env python
"""
precompute_echonext_cache.py
============================
One-time builder for the ECGFMHBEncoder preprocessing cache, EchoNext edition.

Why this exists:
  - The H5 cache builder (precompute_benchmark_cache.py) explicitly skips
    echonext because the loader is .npy-based, not H5. Without a cache, the
    encoder's forward() falls back to live STFT+R-peak preprocessing on the
    main thread — ~80 s/step on H200, GPU 0%.
  - EchoNext records are deterministic (target_length=2500, no random crop),
    so each (split, table_idx) maps to exactly one preprocessing bundle.

Cache layout:
  Same .npz format as the H5 builder (so ECGFMHBEncoder loads it identically).
  Keyed by a synthetic absolute-looking path so cache_key() roundtrips:
      filepath = "/synthetic/echonext/<split>/<table_idx:06d>"
      seg_idx  = 0
  Both the dataset (src/dataset_numpy.py) and this script must agree on that
  format — they import echonext_synthetic_filepath() from the dataset module.

Usage (defaults match the server's paths and existing v4 cache root):
    python scripts/precompute_echonext_cache.py
    python scripts/precompute_echonext_cache.py --workers 64
    python scripts/precompute_echonext_cache.py --splits train      # subset
    python scripts/precompute_echonext_cache.py --force              # rebuild

Idempotent: existing cache files skipped unless --force.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_DEFAULT = os.environ.get("ECG_FM_HB_REPO",
                              "/home/irteam/local-node-d/hbkimi/ecg-fm")
BENCHMARK_DEFAULT = "/home/irteam/local-node-d/hbkimi/benchmark"
CACHE_DEFAULT = os.environ.get(
    "ECG_FM_HB_CACHE",
    "/home/irteam/local-node-d/hbkimi/.cache/ecg_fm_hb_v4",
)
DATA_ROOT_DEFAULT = os.environ.get("ECG_DATA_ROOT", "/home/irteam/ddn-opendata1")


# ──────────────────────────────────────────────────────────────────────────────
# Worker (heavy imports only after pool fork)
# ──────────────────────────────────────────────────────────────────────────────
_WORKER_STATE: dict = {}


def _worker_init(benchmark_repo: str, ecg_fm_repo: str, cache_root: str,
                 max_beats: int, normalize_mode: str, record_mad_scale: float,
                 npy_paths: dict):
    # Single-thread BLAS per worker. With 30+ workers × default OMP we'd spawn
    # 900+ contending threads and tank throughput. Set BEFORE numpy import.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass
    if benchmark_repo not in sys.path:
        sys.path.insert(0, benchmark_repo)
    if ecg_fm_repo not in sys.path:
        sys.path.insert(0, ecg_fm_repo)
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)

    from src.encoders.ecg_fm_hb import (
        _import_pretrain_modules, preprocess_signal,
        cache_path as _cache_path, save_cache, MODEL_FS, MODEL_SEQ_LEN,
    )
    from src.dataset_numpy import echonext_synthetic_filepath
    pp_modules = _import_pretrain_modules(Path(ecg_fm_repo))

    # mmap each split's .npy once per worker
    waveforms_by_split = {}
    for split, p in npy_paths.items():
        if p and Path(p).exists():
            waveforms_by_split[split] = np.load(p, mmap_mode="r")

    _WORKER_STATE.update(
        pp_modules=pp_modules,
        preprocess_signal=preprocess_signal,
        cache_path=_cache_path,
        save_cache=save_cache,
        synth_fp=echonext_synthetic_filepath,
        cache_root=cache_root,
        max_beats=max_beats,
        normalize_mode=normalize_mode,
        record_mad_scale=record_mad_scale,
        model_fs=MODEL_FS,
        model_seq_len=MODEL_SEQ_LEN,
        waveforms=waveforms_by_split,
    )


def _resample_to_model_fs(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return sig.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    return resample_poly(sig, up=up, down=down, axis=-1).astype(np.float32)


def _adjust_length(sig: np.ndarray, target_length: int) -> np.ndarray:
    n_leads, T = sig.shape
    if T == target_length:
        return sig
    if T > target_length:
        return sig[:, :target_length]
    pad = np.zeros((n_leads, target_length - T), dtype=sig.dtype)
    return np.concatenate([sig, pad], axis=-1)


def _read_signal_from_waveforms(wave, table_idx: int, source_fs: int,
                                target_fs: int, target_length: int,
                                n_leads: int, layout: str) -> np.ndarray:
    """Mirror EchoNextDataset._read_signal but operate on the worker-mmapped
    waveforms array. Returns (n_leads, target_length) float32."""
    sig = np.asarray(wave[table_idx]).astype(np.float32)
    if layout == "NHWC":
        # (1, T, C) or (T, C)
        if sig.ndim == 3 and sig.shape[0] == 1:
            sig = sig[0]
        elif sig.ndim == 2:
            pass
        else:
            raise ValueError(f"NHWC unexpected shape: {sig.shape}")
        sig = sig.T  # (C, T)
    elif layout == "NCT":
        if sig.ndim == 3 and sig.shape[0] == 1:
            sig = sig[0]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    if sig.shape[0] != n_leads:
        raise ValueError(f"n_leads mismatch: got {sig.shape[0]}, want {n_leads}")
    if target_fs and target_fs != source_fs:
        from scipy.signal import resample
        target_len_native = int(round(sig.shape[1] * target_fs / source_fs))
        if target_len_native != sig.shape[1]:
            sig = resample(sig, target_len_native, axis=1).astype(np.float32)
    if target_length:
        sig = _adjust_length(sig, target_length)
    sig = np.nan_to_num(sig, nan=0.0)
    return sig


def _process_one(args):
    """Args: (split, table_idx, force, source_fs, target_fs, target_length,
             n_leads, layout)"""
    split, table_idx, force, source_fs, target_fs, target_length, n_leads, layout = args
    state = _WORKER_STATE
    fp = state["synth_fp"](split, table_idx)
    out_path = state["cache_path"](state["cache_root"], fp, 0)
    if out_path.exists() and not force:
        return ("skip_exists", split, table_idx, str(out_path))
    try:
        wave = state["waveforms"].get(split)
        if wave is None:
            return ("fail", split, table_idx, f"no waveforms for split={split}")
        sig = _read_signal_from_waveforms(
            wave, table_idx, source_fs, target_fs, target_length, n_leads, layout,
        )
        # Resample dataset fs → model_fs (500), then fit to model_seq_len (5000)
        sig = _resample_to_model_fs(sig, target_fs or source_fs, state["model_fs"])
        sig = _adjust_length(sig, state["model_seq_len"])

        bundle = state["preprocess_signal"](
            sig, state["pp_modules"],
            max_beats=state["max_beats"],
            normalize_mode=state["normalize_mode"],
            record_mad_scale=state["record_mad_scale"],
        )
        state["save_cache"](out_path, bundle)
        return ("ok", split, table_idx, f"n_valid={bundle['n_valid']}")
    except Exception as e:
        return ("fail", split, table_idx, f"{type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Task discovery
# ──────────────────────────────────────────────────────────────────────────────
def _expand_env(value: str, env: dict) -> str:
    out = value
    for k, v in env.items():
        out = out.replace(f"${{{k}}}", v).replace(f"${k}", v)
    return os.path.expandvars(out)


def _load_echonext_config(benchmark_repo: Path, data_root: str) -> dict:
    cfg_path = benchmark_repo / "configs" / "tasks" / "echonext.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    env = {"ECG_DATA_ROOT": data_root}
    return {
        "metadata_csv":  _expand_env(data["metadata_csv"], env),
        "waveforms":     {k: _expand_env(v, env) for k, v in data["waveforms"].items()},
        "label_cols":    list(data["label_cols"]),
        "split_col":     data.get("split_col", "split"),
        "source_fs":     int(data.get("source_fs", 250)),
        "target_fs":     int(data.get("target_fs", data.get("source_fs", 250))),
        "target_length": int(data.get("target_length", 2500)),
        "n_leads":       int(data.get("n_leads", 12)),
        "layout":        str(data.get("layout", "NHWC")),
    }


def _enumerate_records(cfg: dict, splits: list[str]) -> list[tuple[str, int]]:
    """Return [(split, table_idx), ...] for each requested split, validating
    that the metadata row count matches the .npy length per split."""
    df = pd.read_csv(cfg["metadata_csv"], low_memory=False)
    out = []
    for split in splits:
        npy_path = cfg["waveforms"].get(split)
        if not npy_path or not Path(npy_path).exists():
            print(f"[skip-split] {split}: .npy missing at {npy_path}")
            continue
        n_npy = int(np.load(npy_path, mmap_mode="r").shape[0])
        df_split = df[df[cfg["split_col"]] == split].reset_index(drop=True)
        if len(df_split) != n_npy:
            print(f"[warn] {split}: csv rows={len(df_split)} != npy rows={n_npy}; "
                  "using min(len) — fix your metadata if this isn't expected")
        n = min(len(df_split), n_npy)
        out.extend((split, i) for i in range(n))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-repo", default=BENCHMARK_DEFAULT)
    p.add_argument("--ecg-fm-repo",    default=REPO_DEFAULT)
    p.add_argument("--cache-root",     default=CACHE_DEFAULT)
    p.add_argument("--data-root",      default=DATA_ROOT_DEFAULT)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-beats", type=int, default=30)
    p.add_argument("--normalize-mode", default="record_mad")
    p.add_argument("--record-mad-scale", type=float, default=0.6745)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N (split, idx) tuples (smoke).")
    args = p.parse_args()

    benchmark_repo = Path(args.benchmark_repo).resolve()
    ecg_fm_repo    = Path(args.ecg_fm_repo).resolve()
    cache_root     = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] benchmark_repo : {benchmark_repo}")
    print(f"[info] ecg_fm_repo    : {ecg_fm_repo}")
    print(f"[info] cache_root     : {cache_root}")
    print(f"[info] data_root      : {args.data_root}")
    print(f"[info] splits         : {args.splits}")
    print(f"[info] workers        : {args.workers}")

    cfg = _load_echonext_config(benchmark_repo, args.data_root)
    records = _enumerate_records(cfg, args.splits)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[info] total records  : {len(records)}")

    # Build worker tasks
    tasks = [
        (split, idx, args.force,
         cfg["source_fs"], cfg["target_fs"], cfg["target_length"],
         cfg["n_leads"], cfg["layout"])
        for split, idx in records
    ]

    npy_paths = cfg["waveforms"]

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    failures = []

    ctx = mp.get_context("spawn")  # fork inherits torch CUDA state — avoid
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(benchmark_repo), str(ecg_fm_repo), str(cache_root),
                  args.max_beats, args.normalize_mode, args.record_mad_scale,
                  npy_paths),
    ) as pool:
        for i, (status, split, idx, msg) in enumerate(
            pool.imap_unordered(_process_one, tasks, chunksize=8)
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skip_exists":
                n_skip += 1
            else:
                n_fail += 1
                failures.append((split, idx, msg))
            if (i + 1) % 1000 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (len(tasks) - (i + 1)) / max(rate, 1e-3)
                print(f"[{i+1:>7d}/{len(tasks)}] "
                      f"ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"{rate:.1f} rec/s  eta={eta/60:.1f} min")

    print(f"\n[done] ok={n_ok}  skip={n_skip}  fail={n_fail}  "
          f"elapsed={(time.time()-t0)/60:.1f} min")
    if failures:
        fail_csv = cache_root / "echonext_failures.csv"
        with open(fail_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "table_idx", "error"])
            for s, i_, m in failures:
                w.writerow([s, i_, m])
        print(f"[done] failures.csv : {fail_csv}")


if __name__ == "__main__":
    main()
