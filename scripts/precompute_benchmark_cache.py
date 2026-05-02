#!/usr/bin/env python
"""
precompute_benchmark_cache.py
==============================
One-time builder for the ECGFMHBEncoder preprocessing cache.

The benchmark adapter (src/encoders/ecg_fm_hb.py) is bottlenecked by neurokit2
R-peak detection (~0.4 s/sample). Preprocessing is fully deterministic per H5
file, so we cache it once to <CACHE_ROOT>/<sha1>.npz sidecars and reuse across
every codebook size × eval mode × epoch.

Usage:
    # All paper-canonical tasks, 30 workers, default cache root
    python scripts/precompute_benchmark_cache.py

    # Single task, more workers
    python scripts/precompute_benchmark_cache.py \\
        --tasks ptbxl_super --workers 60

    # Custom cache root and benchmark repo
    ECG_FM_HB_CACHE=/my/cache \\
    python scripts/precompute_benchmark_cache.py \\
        --benchmark-repo /home/irteam/local-node-d/hbkimi/benchmark

The script is idempotent: existing cache files are skipped unless --force is
set. Failures (corrupt H5, no R-peaks, etc.) are recorded in failures.csv but
don't stop the run.

Cache key: sha1(abs_h5_path + "::seg" + seg_idx + "::pp" + version).hexdigest()[:16].
Cached arrays are stored as float16 to halve disk; encoder upcasts to float32
on load. ~0.5 MB per record (PTB-XL 22k → ~10 GB; CODE-15 345k → ~150 GB).
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Optional

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

DEFAULT_TASKS = [
    "ptbxl_super", "ptbxl_sub", "ptbxl_diag", "ptbxl_form", "ptbxl_rhythm",
    "ptbxl_all", "chapman", "chapman_rhythm", "cpsc2018", "cpsc_extra",
    "ningbo", "georgia", "ptb", "code15", "sph_diag", "zzu_pecg",
    # echonext is NumPy (loader_type: echonext_numpy) — skip; encoder always
    # falls back to live preprocessing for that loader.
]


# ──────────────────────────────────────────────────────────────────────────────
# Worker (lazy globals — heavy imports only after pool fork)
# ──────────────────────────────────────────────────────────────────────────────
_WORKER_STATE: dict = {}


def _worker_init(benchmark_repo: str, ecg_fm_repo: str, cache_root: str,
                 max_beats: int, normalize_mode: str, record_mad_scale: float):
    # Single-thread BLAS per worker — w/ 30 workers × default OMP=#cores we'd
    # spawn 900+ contending threads and lose ~5× throughput. Set BEFORE numpy.
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
    # Suppress neurokit2's RuntimeWarnings on bad records
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)

    from src.encoders.ecg_fm_hb import (
        _import_pretrain_modules, preprocess_signal,
        cache_path as _cache_path, save_cache, MODEL_FS, MODEL_SEQ_LEN,
    )
    pp_modules = _import_pretrain_modules(Path(ecg_fm_repo))

    _WORKER_STATE.update(
        pp_modules=pp_modules,
        preprocess_signal=preprocess_signal,
        cache_path=_cache_path,
        save_cache=save_cache,
        cache_root=cache_root,
        max_beats=max_beats,
        normalize_mode=normalize_mode,
        record_mad_scale=record_mad_scale,
        model_fs=MODEL_FS,
        model_seq_len=MODEL_SEQ_LEN,
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


def _process_one(args):
    """
    Args: (abs_h5_path, seg_idx, force)
    Returns: (status, abs_h5_path, message)
        status ∈ {"ok", "skip_exists", "fail"}
    """
    abs_h5, seg_idx, force = args
    state = _WORKER_STATE
    out_path = state["cache_path"](state["cache_root"], abs_h5, seg_idx)
    if out_path.exists() and not force:
        return ("skip_exists", abs_h5, str(out_path))

    try:
        import h5py
        with h5py.File(abs_h5, "r") as f:
            fs = int(f["ECG/metadata"].attrs.get("fs", 500))
            sig = f[f"ECG/segments/{seg_idx}/signal"][()].astype(np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        if sig.ndim != 2 or sig.shape[0] != 12:
            return ("fail", abs_h5, f"bad shape {sig.shape}")

        # Resample H5 fs → 500Hz (model_fs), then fit to 5000 samples (10s)
        sig = _resample_to_model_fs(sig, fs, state["model_fs"])
        sig = _adjust_length(sig, state["model_seq_len"])

        bundle = state["preprocess_signal"](
            sig, state["pp_modules"],
            max_beats=state["max_beats"],
            normalize_mode=state["normalize_mode"],
            record_mad_scale=state["record_mad_scale"],
        )
        state["save_cache"](out_path, bundle)
        return ("ok", abs_h5, f"n_valid={bundle['n_valid']}")
    except Exception as e:
        return ("fail", abs_h5, f"{type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Task discovery
# ──────────────────────────────────────────────────────────────────────────────
def _expand_env(value: str, env: dict) -> str:
    """Cheap ${VAR} expansion using the supplied env dict."""
    out = value
    for k, v in env.items():
        out = out.replace(f"${{{k}}}", v).replace(f"${k}", v)
    return os.path.expandvars(out)


def _collect_h5_paths(
    benchmark_repo: Path,
    task_name: str,
    data_root: str,
) -> list[tuple[str, int]]:
    """Read configs/tasks/<task>.yaml and dedup (abs_path, seg_idx) tuples."""
    cfg_path = benchmark_repo / "configs" / "tasks" / f"{task_name}.yaml"
    if not cfg_path.exists():
        print(f"[warn] {task_name}: config not found at {cfg_path}")
        return []
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    if data.get("loader_type") == "echonext_numpy":
        # NumPy loader: skip (encoder falls back to live for those)
        return []
    env = {"ECG_DATA_ROOT": data_root}
    h5_root = _expand_env(data.get("h5_root", ""), env)
    table_csv = _expand_env(data.get("table_csv", ""), env)
    seg_idx_cfg = data.get("seg_idx", None)
    if not h5_root or not table_csv:
        print(f"[warn] {task_name}: h5_root/table_csv missing")
        return []
    if not os.path.exists(table_csv):
        print(f"[warn] {task_name}: table not found {table_csv}")
        return []

    df = pd.read_csv(table_csv, usecols=["filepath"], low_memory=False)
    paths = []
    if seg_idx_cfg == "all":
        # Need to expand per-record segments. For simplicity here, only seg=0;
        # encoder uses live for any seg!=0 (rare). Comment if you need multi-seg.
        seg_iter = [0]
    else:
        seg_iter = [int(seg_idx_cfg) if seg_idx_cfg is not None else 0]
    for fp in df["filepath"].astype(str).tolist():
        abs_p = os.path.normpath(os.path.join(h5_root, fp))
        for s in seg_iter:
            paths.append((abs_p, s))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Task names (default: all 17 paper-canonical tasks)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--cache-root", default=CACHE_DEFAULT)
    ap.add_argument("--benchmark-repo", default=BENCHMARK_DEFAULT)
    ap.add_argument("--ecg-fm-repo", default=REPO_DEFAULT)
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--max-beats", type=int, default=30)
    ap.add_argument("--normalize", default="record_mad",
                    choices=["record_mad", "zscore", "none"])
    ap.add_argument("--record-mad-scale", type=float, default=5.0)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if cache exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N items (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just print task→count summary; don't compute")
    args = ap.parse_args()

    benchmark_repo = Path(args.benchmark_repo).resolve()
    ecg_fm_repo = Path(args.ecg_fm_repo).resolve()
    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or DEFAULT_TASKS

    # Collect (path, seg_idx) tuples, dedup across tasks
    all_items = []
    seen = set()
    print(f"[scan] tasks={tasks}")
    for t in tasks:
        items = _collect_h5_paths(benchmark_repo, t, args.data_root)
        new = [it for it in items if it not in seen]
        seen.update(items)
        all_items.extend(new)
        print(f"  {t}: +{len(new):,} new ({len(items):,} total)")
    print(f"[scan] unique (h5_path, seg) tuples: {len(all_items):,}")

    if args.limit:
        all_items = all_items[: args.limit]
        print(f"[scan] limited to {len(all_items)} items")

    if args.dry_run:
        return

    if not all_items:
        print("[done] nothing to do")
        return

    pool_args = [(abs_h5, seg, args.force) for abs_h5, seg in all_items]

    failures_path = cache_root / "failures.csv"
    failures_f = open(failures_path, "a", newline="")
    failures_w = csv.writer(failures_f)

    print(f"[run] workers={args.workers}  cache_root={cache_root}")
    t0 = time.time()
    n_ok = n_skip = n_fail = 0

    init_args = (str(benchmark_repo), str(ecg_fm_repo), str(cache_root),
                 args.max_beats, args.normalize, args.record_mad_scale)
    ctx = mp.get_context("spawn")  # avoid fork issues with neurokit2/scipy
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=init_args) as pool:
        for i, (status, path, msg) in enumerate(
            pool.imap_unordered(_process_one, pool_args, chunksize=4),
            start=1,
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skip_exists":
                n_skip += 1
            else:
                n_fail += 1
                failures_w.writerow([path, msg])
            if i % 200 == 0:
                dt = time.time() - t0
                rate = i / dt
                eta = (len(pool_args) - i) / max(rate, 1e-6)
                print(f"  [{i:,}/{len(pool_args):,}] "
                      f"ok={n_ok:,} skip={n_skip:,} fail={n_fail:,} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    failures_f.close()
    dt = time.time() - t0
    print(f"[done] total={len(pool_args):,}  ok={n_ok:,}  skip={n_skip:,}  "
          f"fail={n_fail:,}  elapsed={dt/60:.1f}min")
    if n_fail:
        print(f"[done] failures logged to {failures_path}")


if __name__ == "__main__":
    main()
