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
