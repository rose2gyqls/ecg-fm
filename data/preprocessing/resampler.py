"""
data/preprocessing/resampler.py

Beat segment를 target_length 샘플로 리샘플링.
"""

import numpy as np
from scipy.signal import resample, resample_poly
from math import gcd


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int = 500) -> np.ndarray:
    """
    (12, T) 신호를 fs_in → fs_out 으로 리샘플. 정수비일 때 polyphase 사용.
    """
    if fs_in == fs_out:
        return signal.astype(np.float32)
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    out = resample_poly(signal, up=up, down=down, axis=-1)
    return out.astype(np.float32)


def resample_beat(beat: np.ndarray, target_length: int = 256) -> np.ndarray:
    """
    Args:
        beat : (..., W) — last dim is time
    Returns:
        resampled : (..., target_length)
    """
    if beat.shape[-1] == target_length:
        return beat.astype(np.float32)
    return resample(beat, target_length, axis=-1).astype(np.float32)


def normalize_beat(beat: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Beat-wise normalization."""
    if method == "zscore":
        mu  = beat.mean(axis=-1, keepdims=True)
        std = beat.std(axis=-1, keepdims=True) + 1e-8
        return (beat - mu) / std
    elif method == "minmax":
        mn = beat.min(axis=-1, keepdims=True)
        mx = beat.max(axis=-1, keepdims=True)
        return (beat - mn) / (mx - mn + 1e-8)
    return beat


# ── Record-level robust normalization (preserves inter-lead amplitude) ──────
# Per-record (median, robust-scale) once for the whole (12, T) signal so that
# V1 vs V6 amplitude differences survive into the codebook input. Per-beat
# z-score erases that.
#
# Note on the scale estimator: classical MAD (median of |sig − median|)
# degenerates to ~0 on raw ECG because the long isoelectric baseline puts
# most samples near the median (heedb is also heavily quantized via float16
# storage). We use the 75th percentile of |sig − median| instead — still
# robust to outliers, but non-degenerate as long as the signal has any
# diagnostic content. For a clean Gaussian, p75/0.6745 ≈ σ, so a beat with
# QRS amplitude ~1 mV ends up at ~`1 mV / (scale · p75)` ≈ a few units.

def compute_record_norm_stats(
    signal: np.ndarray, eps: float = 1e-6, percentile: float = 75.0
) -> tuple[float, float]:
    """Global median and percentile-based robust scale over (lead, time)."""
    median = float(np.median(signal))
    scale = float(np.percentile(np.abs(signal - median), percentile)) + eps
    return median, scale


def apply_record_norm(
    x: np.ndarray, median: float, robust_scale: float, scale: float = 5.0
) -> np.ndarray:
    """Apply record-level (median, robust_scale)·scale normalization.

    `robust_scale` is the value returned by `compute_record_norm_stats`
    (75th percentile of |sig − median| by default). `scale` is an additional
    multiplier so that the normalized output sits in roughly ±1 for typical
    diagnostic content.
    """
    return ((x - median) / (scale * robust_scale)).astype(np.float32)
