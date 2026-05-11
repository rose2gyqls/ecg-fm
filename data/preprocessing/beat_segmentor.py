from __future__ import annotations
import numpy as np
from typing import Optional
import warnings

def detect_rpeaks(signal: np.ndarray, fs: int, method: str = "neurokit") -> np.ndarray:
    if method == "neurokit":
        try:
            import neurokit2 as nk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
            return np.array(info["ECG_R_Peaks"], dtype=int)
        except ImportError:
            warnings.warn("neurokit2 not installed, falling back to wfdb")
            method = "wfdb"
    if method == "wfdb":
        try:
            import wfdb.processing as wfp
            rpeaks = wfp.qrs_detect(signal, fs=fs)
            return np.array(rpeaks, dtype=int)
        except ImportError:
            raise ImportError("wfdb not installed. pip install wfdb")
    raise ValueError(f"Unknown method: {method}")

def validate_rpeaks_boundary(
    rpeaks: np.ndarray, T: int, before: int, after: int
) -> np.ndarray:
    if len(rpeaks) == 0:
        return rpeaks
    mask = (rpeaks >= before) & (rpeaks + after <= T)
    return rpeaks[mask]

def validate_rpeaks_local_max(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    window: int = 10,
    max_shift: int = 8,
) -> np.ndarray:
    if len(rpeaks) == 0:
        return rpeaks
    T = signal.shape[-1]
    abs_sig = np.abs(signal)
    keep = np.zeros(len(rpeaks), dtype=bool)
    for i, r in enumerate(rpeaks):
        lo = max(0, int(r) - window)
        hi = min(T, int(r) + window + 1)
        if hi <= lo:
            continue
        local_max_pos = lo + int(np.argmax(abs_sig[lo:hi]))
        if abs(local_max_pos - int(r)) <= max_shift:
            keep[i] = True
    return rpeaks[keep]

def compute_rr_features(rpeaks: np.ndarray, fs: int) -> list[dict]:
    n = len(rpeaks)
    rr = np.diff(rpeaks) / fs
    median_rr = float(np.median(rr)) if len(rr) > 0 else 0.8
    features = []
    for i in range(n):
        prev_rr = float(rr[i - 1]) if i > 0 else median_rr
        next_rr = float(rr[i])     if i < n - 1 else median_rr
        features.append({
            "prev_rr":   prev_rr,
            "next_rr":   next_rr,
            "median_rr": median_rr,
        })
    return features

def extract_beats(
    ecg: np.ndarray,
    rpeaks: np.ndarray,
    fs: int,
    before_ms: int = 200,
    after_ms: int = 400,
    pad_value: float = 0.0,
) -> list[np.ndarray]:
    before = int(fs * before_ms / 1000)
    after  = int(fs * after_ms  / 1000)
    T = ecg.shape[-1]
    beats = []
    for r in rpeaks:
        start = r - before
        end   = r + after
        if ecg.ndim == 1:
            segment = np.full(before + after, pad_value, dtype=np.float32)
            src_s = max(start, 0)
            src_e = min(end, T)
            dst_s = src_s - start
            segment[dst_s: dst_s + (src_e - src_s)] = ecg[src_s:src_e]
        else:
            L = ecg.shape[0]
            segment = np.full((L, before + after), pad_value, dtype=np.float32)
            src_s = max(start, 0)
            src_e = min(end, T)
            dst_s = src_s - start
            segment[:, dst_s: dst_s + (src_e - src_s)] = ecg[:, src_s:src_e]
        beats.append(segment)
    return beats

def compute_qrs_intervals(
    beats: np.ndarray,
    before_ms: int = 200,
    after_ms: int = 400,
    beat_length: int = 256,
    q_window_ms: int = 50,
    s_window_ms: int = 80,
) -> np.ndarray:
    total_ms      = before_ms + after_ms
    ms_per_sample = total_ms / beat_length
    r_pos         = int(round(before_ms / total_ms * beat_length))
    q_span = max(1, int(round(q_window_ms / ms_per_sample)))
    s_span = max(1, int(round(s_window_ms / ms_per_sample)))
    q_lo   = max(0, r_pos - q_span)
    q_hi   = r_pos
    s_lo   = r_pos + 1
    s_hi   = min(beat_length, r_pos + 1 + s_span)
    if q_hi <= q_lo or s_hi <= s_lo:
        return np.zeros(beats.shape[:2] + (2,), dtype=np.float32)
    q_seg = beats[..., q_lo:q_hi]
    s_seg = beats[..., s_lo:s_hi]
    q_idx = np.argmin(q_seg, axis=-1) + q_lo
    s_idx = np.argmin(s_seg, axis=-1) + s_lo
    qr_sec = ((r_pos - q_idx) * ms_per_sample / 1000.0).astype(np.float32)
    rs_sec = ((s_idx - r_pos) * ms_per_sample / 1000.0).astype(np.float32)
    return np.stack([qr_sec, rs_sec], axis=-1)

def flat_beat_mask(
    beats: np.ndarray,
    ptp_min: float = 0.1,
    std_min: float = 0.01,
) -> np.ndarray:
    ptp = beats.max(axis=-1) - beats.min(axis=-1)
    std = beats.std(axis=-1)
    return (ptp < ptp_min) | (std < std_min)
LEAD_NAMES = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
LEAD_II_INDEX = 1

def process_ecg_record(
    ecg: np.ndarray,
    fs: int,
    ref_lead_idx: int = 1,
    before_ms: int = 200,
    after_ms: int = 400,
    rpeak_method: str = "neurokit",
) -> Optional[dict]:
    assert ecg.ndim == 2 and ecg.shape[0] == 12, f"Expected (12, T), got {ecg.shape}"
    ref_signal = ecg[ref_lead_idx]
    try:
        rpeaks = detect_rpeaks(ref_signal, fs, method=rpeak_method)
    except Exception as e:
        warnings.warn(f"R-peak detection failed: {e}")
        return None
    if len(rpeaks) < 2:
        return None
    beats_per_lead = []
    for l in range(12):
        beats_l = extract_beats(ecg[l], rpeaks, fs, before_ms, after_ms)
        beats_per_lead.append(beats_l)
    n_beats = len(rpeaks)
    W = beats_per_lead[0][0].shape[0]
    beats_arr = np.zeros((n_beats, 12, W), dtype=np.float32)
    for l in range(12):
        for b in range(n_beats):
            beats_arr[b, l, :] = beats_per_lead[l][b]
    rr_feats = compute_rr_features(rpeaks, fs)
    return {
        "beats":    beats_arr,
        "rr_feats": rr_feats,
        "rpeaks":   rpeaks,
        "n_beats":  n_beats,
    }
