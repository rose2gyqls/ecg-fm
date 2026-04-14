"""
data/preprocessing/heedb_io.py

HEEDB(.h5) 레코드 하나를 읽어 (signal, fs, rpeaks, meta) 반환.
포맷 사양:
  root.attrs["beat_ext_method"]
  ECG/metadata.attrs["fs"], ["sig_len"]
  ECG/segments/0/signal                  (12, T) float16
  ECG/segments/0/beat_annotation/sample  (N,)   int16   (존재 시)
채널 순서: I II III V1 V2 V3 V4 V5 V6 aVF aVL aVR
"""

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
    """
    반환 dict:
        signal : (12, T) float32
        fs     : int
        rpeaks : Optional[np.ndarray] (N,) int  — beat_annotation 있을 때만
        fs_sig_name : list[str]
    레코드가 비정상이면 None 반환.
    """
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

        signal = seg_grp["signal"][()].astype(np.float32)   # (12, T)
        if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < fs:
            return None

        # zero-lead 필터
        if np.any(np.all(signal == 0, axis=1)):
            return None
        # NaN -> 0 (행 단위 제로 보정은 후속 단계에서)
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
    """
    sig_name이 HEEDB_LEAD_ORDER 와 다르면 재정렬. 일치하지 않으면 None.
    """
    if sig_name == HEEDB_LEAD_ORDER:
        return signal
    try:
        idx = [sig_name.index(n) for n in HEEDB_LEAD_ORDER]
    except ValueError:
        return None
    return signal[idx]
