"""
HEEDB 전체 H5 파일 리스트를 train/val로 분할해서 저장.
재현 가능한 seed 기반 셔플.
"""
import glob
import os
import random
import sys

HEEDB_ROOT = "/home/irteam/ddn-opendata1/h5/heedb"
OUT_DIR    = "/home/irteam/local-node-d/hbkimi/ecg-fm/file_lists"
SEED       = 42
VAL_COUNT  = 10_000

TRAIN_OUT = os.path.join(OUT_DIR, "train_files_full.txt")
VAL_OUT   = os.path.join(OUT_DIR, "val_files_full.txt")


def main():
    print(f"[scan] {HEEDB_ROOT}", flush=True)
    files = sorted(glob.glob(os.path.join(HEEDB_ROOT, "**", "*.h5"),
                             recursive=True))
    print(f"[scan] found {len(files):,}", flush=True)
    if len(files) < VAL_COUNT * 2:
        sys.exit(f"too few files: {len(files)}")

    rng = random.Random(SEED)
    rng.shuffle(files)

    val   = files[:VAL_COUNT]
    train = files[VAL_COUNT:]

    with open(TRAIN_OUT, "w") as f:
        f.write("\n".join(train) + "\n")
    with open(VAL_OUT, "w") as f:
        f.write("\n".join(val) + "\n")

    print(f"[write] train={len(train):,} → {TRAIN_OUT}")
    print(f"[write] val  ={len(val):,} → {VAL_OUT}")


if __name__ == "__main__":
    main()
