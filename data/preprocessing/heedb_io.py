from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import h5py
HEEDB_LEAD_ORDER = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]

def load_heedb_record(
    path: str,
    seg_idx: int = 0,
    load_rpeaks: bool = True,
) -> Optional[dict]:
    with h5py.File(path, "r") as f:
        if "ECG" not in f:
            return None
        meta = f["ECG/metadata"]
        fs = int(meta.attrs["fs"])
        sig_name = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in meta["sig_name"][()]]
        seg_grp = f.get(f"ECG/segments/{seg_idx}")
        if seg_grp is None or "signal" not in seg_grp:
            return None
        signal = seg_grp["signal"][()].astype(np.float32)
        if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < fs:
            return None
        if np.any(np.all(signal == 0, axis=1)):
            return None
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        rpeaks = None
        if load_rpeaks and "beat_annotation" in seg_grp:
            ba = seg_grp["beat_annotation"]
            if "sample" in ba:
                rpeaks = ba["sample"][()].astype(np.int64)
    return {
        "signal":   signal,
        "fs":       fs,
        "rpeaks":   rpeaks,
        "sig_name": sig_name,
    }

def align_to_heedb_order(signal: np.ndarray, sig_name: list) -> Optional[np.ndarray]:
    if sig_name == HEEDB_LEAD_ORDER:
        return signal
    try:
        idx = [sig_name.index(n) for n in HEEDB_LEAD_ORDER]
    except ValueError:
        return None
    return signal[idx]
