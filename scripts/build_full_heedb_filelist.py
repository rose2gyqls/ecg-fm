"""Build reproducible train/validation file lists for HEEDB H5 records."""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

DEFAULT_OUT_DIR = "file_lists"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heedb-root", default=os.environ.get("HEEDB_ROOT"),
                        help="Root directory containing HEEDB .h5 files. "
                             "Can also be set via HEEDB_ROOT.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=10_000)
    args = parser.parse_args()

    if not args.heedb_root:
        sys.exit("Set --heedb-root or HEEDB_ROOT.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_files_full.txt"
    val_out = out_dir / "val_files_full.txt"

    print(f"[scan] {args.heedb_root}", flush=True)
    files = sorted(glob.glob(os.path.join(args.heedb_root, "**", "*.h5"),
                             recursive=True))
    print(f"[scan] found {len(files):,}", flush=True)
    if len(files) < args.val_count * 2:
        sys.exit(f"too few files: {len(files)}")

    rng = random.Random(args.seed)
    rng.shuffle(files)

    val = files[:args.val_count]
    train = files[args.val_count:]

    with train_out.open("w") as f:
        f.write("\n".join(train) + "\n")
    with val_out.open("w") as f:
        f.write("\n".join(val) + "\n")

    print(f"[write] train={len(train):,} -> {train_out}")
    print(f"[write] val  ={len(val):,} -> {val_out}")


if __name__ == "__main__":
    main()
