from __future__ import annotations
import argparse, glob, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.preprocessing.heedb_io       import load_heedb_record, align_to_heedb_order, HEEDB_LEAD_ORDER
from data.preprocessing.beat_segmentor import detect_rpeaks, extract_beats, LEAD_II_INDEX
from data.preprocessing.resampler      import resample_signal, resample_beat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--target_fs", type=int, default=500)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.h5"), recursive=True))
    assert files, f"No h5 files under {args.data_dir}"
    random.shuffle(files)
    print(f"Total h5 files: {len(files):,}")
    checked = 0
    for p in files:
        rec = load_heedb_record(p, load_rpeaks=True)
        if rec is None:
            continue
        sig = align_to_heedb_order(rec["signal"], rec["sig_name"])
        if sig is None:
            print(f"  [skip] sig_name mismatch: {rec['sig_name']}")
            continue
        fs_in = rec["fs"]
        print(f"\n=== {os.path.basename(p)} ===")
        print(f"  fs_in={fs_in}  shape={sig.shape}  leads_ok={rec['sig_name']==HEEDB_LEAD_ORDER}")
        if fs_in != args.target_fs:
            sig_rs = resample_signal(sig, fs_in, args.target_fs)
            print(f"  resampled -> shape={sig_rs.shape} (fs={args.target_fs})")
        else:
            sig_rs = sig
        fs = args.target_fs
        rp_src = "heedb"
        rpeaks = rec["rpeaks"]
        if rpeaks is not None and fs_in != args.target_fs:
            rpeaks = (rpeaks.astype(np.float64) * args.target_fs / fs_in).astype(np.int64)
        if rpeaks is None or len(rpeaks) < 2:
            rpeaks = detect_rpeaks(sig_rs[LEAD_II_INDEX], fs, method="neurokit")
            rp_src = "neurokit"
        print(f"  rpeaks: n={len(rpeaks)}  source={rp_src}")
        beats = extract_beats(sig_rs, rpeaks, fs, before_ms=200, after_ms=400)
        beats = np.stack(beats, axis=0)
        print(f"  beats: shape={beats.shape}  dtype={beats.dtype}")
        N, L, W = beats.shape
        flat = beats.reshape(N*L, W)
        flat_rs = resample_beat(flat, 256)
        print(f"  flat for codebook: {flat_rs.shape} (should be {N*12}, 256)")
        checked += 1
        if checked >= args.n:
            break
    print(f"\nOK: sanity-checked {checked} records.")

if __name__ == "__main__":
    main()
