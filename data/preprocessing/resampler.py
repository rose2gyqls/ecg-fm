import numpy as np
from scipy.signal import resample, resample_poly
from math import gcd

def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int = 500) -> np.ndarray:
    if fs_in == fs_out:
        return signal.astype(np.float32)
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    out = resample_poly(signal, up=up, down=down, axis=-1)
    return out.astype(np.float32)

def resample_beat(beat: np.ndarray, target_length: int = 256) -> np.ndarray:
    if beat.shape[-1] == target_length:
        return beat.astype(np.float32)
    return resample(beat, target_length, axis=-1).astype(np.float32)

def normalize_beat(beat: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mu  = beat.mean(axis=-1, keepdims=True)
        std = beat.std(axis=-1, keepdims=True) + 1e-8
        return (beat - mu) / std
    elif method == "minmax":
        mn = beat.min(axis=-1, keepdims=True)
        mx = beat.max(axis=-1, keepdims=True)
        return (beat - mn) / (mx - mn + 1e-8)
    return beat

def compute_record_norm_stats(
    signal: np.ndarray,
    eps: float = 1e-6,
    percentile: float = 75.0,
    min_scale: float = 0.05,
) -> tuple[float, float]:
    median = float(np.median(signal))
    if signal.ndim >= 2:
        per_lead = np.percentile(np.abs(signal - median), percentile, axis=-1)
        scale = float(np.median(per_lead)) + eps
    else:
        scale = float(np.percentile(np.abs(signal - median), percentile)) + eps
    scale = max(scale, min_scale)
    return median, scale

def apply_record_norm(
    x: np.ndarray,
    median: float,
    robust_scale: float,
    scale: float = 5.0,
    clip: float | None = None,
) -> np.ndarray:
    out = ((x - median) / (scale * robust_scale)).astype(np.float32)
    if clip is not None:
        np.clip(out, -float(clip), float(clip), out=out)
    return out
