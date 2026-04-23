"""
data/preprocessing/beat_segmentor.py

R-peak 검출 및 리드별 beat segment 추출.
wfdb, neurokit2 기반. 필요에 따라 pan-tompkins 직접 구현으로 교체 가능.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import warnings


# ──────────────────────────────────────────────────────────────────────────────
# R-peak detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_rpeaks(signal: np.ndarray, fs: int, method: str = "neurokit") -> np.ndarray:
    """
    Lead II (또는 임의 1D 신호)에서 R-peak 인덱스를 반환.

    Args:
        signal : (T,) 1D ECG signal
        fs     : sampling frequency
        method : "neurokit" | "wfdb" | "pantompkins"

    Returns:
        rpeaks : (N,) int array of R-peak sample indices
    """
    if method == "neurokit":
        try:
            import neurokit2 as nk
            # 저품질 신호에서 neurokit 내부가 np.mean/empty slice RuntimeWarning을
            # 대량으로 뱉어서 로그를 오염시킴 → 검출 실패는 len(rpeaks)<2로 거르므로
            # 이 구간의 RuntimeWarning은 안전하게 무시.
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


# ──────────────────────────────────────────────────────────────────────────────
# R-peak validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_rpeaks_boundary(
    rpeaks: np.ndarray, T: int, before: int, after: int
) -> np.ndarray:
    """before 샘플 앞/after 샘플 뒤가 완전히 신호 안에 들어오는 R-peak만 유지."""
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
    """R-peak이 ±window 안에서 |signal| argmax와 max_shift 이내에 있는지 확인."""
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


# ──────────────────────────────────────────────────────────────────────────────
# RR interval features
# ──────────────────────────────────────────────────────────────────────────────

def compute_rr_features(rpeaks: np.ndarray, fs: int) -> list[dict]:
    """
    각 비트에 대해 prev_rr, next_rr, median_rr (초 단위) dict 리스트 반환.
    첫 번째/마지막 비트의 prev/next_rr는 median으로 padding.
    """
    n = len(rpeaks)
    rr = np.diff(rpeaks) / fs                          # (N-1,) in seconds
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


# ──────────────────────────────────────────────────────────────────────────────
# Beat extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_beats(
    ecg: np.ndarray,
    rpeaks: np.ndarray,
    fs: int,
    before_ms: int = 200,
    after_ms: int = 400,
    pad_value: float = 0.0,
) -> list[np.ndarray]:
    """
    rpeaks 위치를 중심으로 각 beat segment를 추출.

    Args:
        ecg       : (T,) or (L, T) — single or multi-lead
        rpeaks    : R-peak sample indices
        fs        : sampling rate
        before_ms : R-peak 앞으로 포함할 ms
        after_ms  : R-peak 뒤로 포함할 ms
        pad_value : 경계 초과 시 패딩 값

    Returns:
        beats : list of (W,) or (L, W) arrays  (W = before+after samples)
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Fiducial intervals (Q-R, R-S) — pretrain regression target
# ──────────────────────────────────────────────────────────────────────────────

def compute_qrs_intervals(
    beats: np.ndarray,
    before_ms: int = 200,
    after_ms: int = 400,
    beat_length: int = 256,
    q_window_ms: int = 50,
    s_window_ms: int = 80,
) -> np.ndarray:
    """
    Resampled beat 배열 (N, L, W) 에서 각 비트/리드의 Q-R, R-S 간격을 추정.

    R은 extract_beats가 before_ms 지점에 정렬해 두었으므로
        r_pos = round(before_ms / (before_ms+after_ms) * beat_length)
    이다 (default: 200/600*256 = 85).

    Q = argmin of signal in [r_pos - q_window, r_pos) — R 직전 50ms 창의 최저점
    S = argmin of signal in (r_pos, r_pos + s_window] — R 직후 80ms 창의 최저점

    zscore-normalize된 beat에서도 정상 QRS의 경우 Q/S가 국소 최저점이므로
    충분히 안정적. 병적 QRS(RBBB 등)는 정의 자체가 흐려지지만, regression
    target으로는 기대값이 유지되므로 사용 가능.

    Args:
        beats       : (N, L, W) float32 — 정규화된 resampled beats
        before_ms   : extract_beats에 쓴 값과 동일해야 함
        after_ms    : extract_beats에 쓴 값과 동일해야 함
        beat_length : W와 동일
        q_window_ms : Q 탐색창 (R 직전)
        s_window_ms : S 탐색창 (R 직후)

    Returns:
        fid : (N, L, 2) float32 — [qr_sec, rs_sec]
    """
    total_ms      = before_ms + after_ms
    ms_per_sample = total_ms / beat_length
    r_pos         = int(round(before_ms / total_ms * beat_length))

    q_span = max(1, int(round(q_window_ms / ms_per_sample)))
    s_span = max(1, int(round(s_window_ms / ms_per_sample)))
    q_lo   = max(0, r_pos - q_span)
    q_hi   = r_pos                           # [q_lo, q_hi)
    s_lo   = r_pos + 1
    s_hi   = min(beat_length, r_pos + 1 + s_span)  # [s_lo, s_hi)

    # 잘못된 window 구성 방어
    if q_hi <= q_lo or s_hi <= s_lo:
        return np.zeros(beats.shape[:2] + (2,), dtype=np.float32)

    q_seg = beats[..., q_lo:q_hi]            # (N, L, q_span)
    s_seg = beats[..., s_lo:s_hi]            # (N, L, s_span)

    q_idx = np.argmin(q_seg, axis=-1) + q_lo # (N, L)
    s_idx = np.argmin(s_seg, axis=-1) + s_lo # (N, L)

    qr_sec = ((r_pos - q_idx) * ms_per_sample / 1000.0).astype(np.float32)
    rs_sec = ((s_idx - r_pos) * ms_per_sample / 1000.0).astype(np.float32)
    return np.stack([qr_sec, rs_sec], axis=-1)  # (N, L, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Noise / flat-beat filter
# ──────────────────────────────────────────────────────────────────────────────

def flat_beat_mask(
    beats: np.ndarray,
    ptp_min: float = 0.1,
    std_min: float = 0.01,
) -> np.ndarray:
    """
    Raw 단위(mV)의 beat 배열에 대해 flat/noise 마스크를 반환.
    True = flat (drop 대상).

    Args:
        beats : (..., W) — 임의 앞차원 + 시간
    Returns:
        mask  : (...,) bool — beats[...,W]에서 W를 제거한 shape
    """
    ptp = beats.max(axis=-1) - beats.min(axis=-1)
    std = beats.std(axis=-1)
    return (ptp < ptp_min) | (std < std_min)


# ──────────────────────────────────────────────────────────────────────────────
# High-level: process one 12-lead ECG record
# ──────────────────────────────────────────────────────────────────────────────

# HEEDB 채널 순서 (signal[l] 인덱스와 일치)
LEAD_NAMES = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
LEAD_II_INDEX = 1

def process_ecg_record(
    ecg: np.ndarray,
    fs: int,
    ref_lead_idx: int = 1,             # Lead II for R-peak detection
    before_ms: int = 200,
    after_ms: int = 400,
    rpeak_method: str = "neurokit",
) -> Optional[dict]:
    """
    12-lead ECG array (12, T) 를 입력받아
    beat segment, RR features, lead info를 반환하는 메인 함수.

    Returns dict with keys:
        beats     : (N_beats, 12, W) float32
        rr_feats  : list of N_beats dicts
        rpeaks    : (N_beats,) int
        n_beats   : int
    """
    assert ecg.ndim == 2 and ecg.shape[0] == 12, \
        f"Expected (12, T), got {ecg.shape}"

    # R-peak detection on reference lead
    ref_signal = ecg[ref_lead_idx]
    try:
        rpeaks = detect_rpeaks(ref_signal, fs, method=rpeak_method)
    except Exception as e:
        warnings.warn(f"R-peak detection failed: {e}")
        return None

    if len(rpeaks) < 2:
        return None

    # Per-beat extraction
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
        "beats":    beats_arr,    # (N, 12, W)
        "rr_feats": rr_feats,     # list of N dicts
        "rpeaks":   rpeaks,       # (N,)
        "n_beats":  n_beats,
    }
