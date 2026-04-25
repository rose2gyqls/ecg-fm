"""
training/tokenizer/losses.py

L_total = L_rec + α·L_vq + β·L_fid + γ·L_spec

  L_rec  : MSE(x, x̂)
  L_vq   : commitment (EMA) or commitment + codebook (non-EMA)
  L_fid  : (a) gradient proxy ‖∇x − ∇x̂‖²  ⟶ QRS 등 변화 구간 보존
           (b) point-wise weighted MSE (QRS 가우시안 가중치) — config 토글
  L_spec : multi-scale STFT magnitude L1 — 시간축 MSE만으론 morphology가
           부드럽게 깎이는 문제를 보완
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Iterable, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction
# ──────────────────────────────────────────────────────────────────────────────
def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """L_rec: MSE between original and reconstructed beat."""
    return F.mse_loss(x_hat, x)


# ──────────────────────────────────────────────────────────────────────────────
# Fiducial — gradient proxy (default)
# ──────────────────────────────────────────────────────────────────────────────
def gradient_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    ‖∇x − ∇x̂‖²   — finite difference along time axis (last dim).
    QRS / P / T 의 급격한 기울기 보존.
    """
    dx = x[..., 1:] - x[..., :-1]
    dx_hat = x_hat[..., 1:] - x_hat[..., :-1]
    return F.mse_loss(dx_hat, dx)


# ──────────────────────────────────────────────────────────────────────────────
# Fiducial — point-wise weighted MSE (QRS-focused)
# ──────────────────────────────────────────────────────────────────────────────
def make_qrs_weight_map(
    beat_length: int = 256,
    r_pos: int = 85,
    sigma: float = 20.0,
    base_weight: float = 1.0,
    peak_weight: float = 3.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Gaussian bump weight map peaked at R-peak position.

    Default geometry assumes:
      before_ms=200, after_ms=400 → R at 1/3 of the resampled 256 = 85.

    sigma=20 (~40 sample full width) covers Q–R–S 윈도우(±80ms 정도).
    Returns: (beat_length,) tensor.
    """
    t = torch.arange(beat_length, dtype=dtype, device=device)
    g = torch.exp(-0.5 * ((t - r_pos) / sigma) ** 2)
    return base_weight + (peak_weight - base_weight) * g


def weighted_mse_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Point-wise weighted MSE.
    weights: (W,) broadcast 가능한 형태. mean으로 정규화돼 있다고 보고 그대로 사용.
    """
    # broadcast: weights (W,) → (1, 1, W)
    w = weights.view(1, 1, -1) if weights.dim() == 1 else weights
    return ((x_hat - x) ** 2 * w).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Spectral — multi-scale STFT magnitude L1
# ──────────────────────────────────────────────────────────────────────────────
# torch.hann_window를 매 step 만들지 않도록 캐시.
_window_cache: dict[tuple[int, str, str], torch.Tensor] = {}


def _hann_window(n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n_fft, str(device), str(dtype))
    w = _window_cache.get(key)
    if w is None or w.device != device:
        w = torch.hann_window(n_fft, device=device, dtype=dtype)
        _window_cache[key] = w
    return w


def multiscale_stft_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    n_ffts: Iterable[int] = (32, 64, 128),
) -> torch.Tensor:
    """
    Multi-scale STFT magnitude L1.

    Args:
        x, x_hat : (B, 1, W) — z-scored beat waveforms.
                   beat_length=256, fs=500Hz라 n_fft={32,64,128} (= 64ms~256ms 윈도우)가
                   QRS 시간 스케일과 잘 맞는다.
    """
    # squeeze channel dim
    if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)

    total = x.new_zeros(())
    n_scales = 0
    for n_fft in n_ffts:
        if n_fft >= x.shape[-1]:
            # beat_length보다 큰 n_fft는 의미 없음 — 그냥 skip
            continue
        hop = max(n_fft // 4, 1)
        win = _hann_window(n_fft, x.device, x.dtype)
        Xs = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True, center=True,
        )
        Xs_hat = torch.stft(
            x_hat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True, center=True,
        )
        total = total + F.l1_loss(Xs_hat.abs(), Xs.abs())
        n_scales += 1

    if n_scales == 0:
        return x.new_zeros(())
    return total / n_scales


# ──────────────────────────────────────────────────────────────────────────────
# Total
# ──────────────────────────────────────────────────────────────────────────────
def total_vqvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    loss_vq: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.0,
    use_gradient_loss: bool = True,
    fiducial_weights: Optional[torch.Tensor] = None,
    spec_n_ffts: Iterable[int] = (32, 64, 128),
) -> dict:
    """
    Returns dict with individual + total losses.

    Notes:
      - `use_gradient_loss=True` → L_fid = gradient_loss (시간축 미분 MSE)
        `False` + fiducial_weights 제공 → L_fid = weighted_mse_loss(QRS 가중)
        `False` + weights=None → L_fid = plain MSE (= L_rec과 동일, 사실상 비활성)
      - gamma=0이면 spectral loss 계산 자체를 건너뛰어 cost 0.
    """
    l_rec = reconstruction_loss(x, x_hat)

    if use_gradient_loss:
        l_fid = gradient_loss(x, x_hat)
    elif fiducial_weights is not None:
        l_fid = weighted_mse_loss(x, x_hat, fiducial_weights)
    else:
        l_fid = l_rec.detach() * 0  # no-op

    if gamma > 0:
        l_spec = multiscale_stft_loss(x, x_hat, n_ffts=spec_n_ffts)
    else:
        l_spec = x.new_zeros(())

    total = l_rec + alpha * loss_vq + beta * l_fid + gamma * l_spec

    return {
        "loss":      total,
        "loss_rec":  l_rec,
        "loss_vq":   loss_vq,
        "loss_fid":  l_fid,
        "loss_spec": l_spec,
    }
