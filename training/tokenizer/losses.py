"""
VQ-VAE training losses.

L_total = L_rec + alpha * L_vq + beta * L_fid + gamma * L_spec

  L_rec  : MSE in time domain.
  L_vq   : commitment (EMA mode) or commitment + codebook (non-EMA mode);
           computed inside VQCodebook and passed in.
  L_fid  : either gradient proxy ||dx - dx_hat||^2 (default) to preserve
           QRS-like fast transitions, or a point-wise weighted MSE peaked
           around the R position.
  L_spec : multi-scale STFT magnitude L1, complementing the time-domain
           MSE so morphology stays sharp instead of being smoothed out.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Iterable, Optional


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------
def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


# -----------------------------------------------------------------------------
# Fiducial proxies
# -----------------------------------------------------------------------------
def gradient_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """||dx - dx_hat||^2 with dx as a finite difference along the last axis."""
    dx = x[..., 1:] - x[..., :-1]
    dx_hat = x_hat[..., 1:] - x_hat[..., :-1]
    return F.mse_loss(dx_hat, dx)


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
    Gaussian bump centered at the R-peak position.

    Default geometry assumes before_ms=200, after_ms=400, beat_length=256
    -> r_pos = round(200/600 * 256) = 85. sigma=20 covers roughly
    +-80 ms around R, i.e. the Q-R-S window.
    """
    t = torch.arange(beat_length, dtype=dtype, device=device)
    g = torch.exp(-0.5 * ((t - r_pos) / sigma) ** 2)
    return base_weight + (peak_weight - base_weight) * g


def weighted_mse_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MSE multiplied by `weights` (broadcast-compatible)."""
    w = weights.view(1, 1, -1) if weights.dim() == 1 else weights
    return ((x_hat - x) ** 2 * w).mean()


# -----------------------------------------------------------------------------
# Multi-scale STFT magnitude L1
# -----------------------------------------------------------------------------
# Cache hann windows so we don't reallocate one per step.
_window_cache: dict[tuple[int, str, str], torch.Tensor] = {}


def _hann_window(n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n_fft, str(device), str(dtype))
    w = _window_cache.get(key)
    if w is None:
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
        x, x_hat: (B, 1, W) z-scored beat waveforms (W=256, fs=500 Hz).
                  n_fft in {32, 64, 128} corresponds to 64ms / 128ms / 256ms
                  windows, which bracket the QRS time scale.

    n_fft values >= W are skipped silently.
    """
    if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)

    total = x.new_zeros(())
    n_scales = 0
    for n_fft in n_ffts:
        if n_fft >= x.shape[-1]:
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

    return total / n_scales if n_scales > 0 else x.new_zeros(())


# -----------------------------------------------------------------------------
# Total loss
# -----------------------------------------------------------------------------
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
    Combine the four reconstruction-side terms with the codebook commitment.

    Fiducial behavior:
        use_gradient_loss=True             -> L_fid = gradient_loss(x, x_hat)
        use_gradient_loss=False, weights!=None -> L_fid = weighted MSE around R
        use_gradient_loss=False, weights=None -> L_fid disabled (zero)

    gamma=0 short-circuits the spectral loss entirely.
    """
    l_rec = reconstruction_loss(x, x_hat)

    if use_gradient_loss:
        l_fid = gradient_loss(x, x_hat)
    elif fiducial_weights is not None:
        l_fid = weighted_mse_loss(x, x_hat, fiducial_weights)
    else:
        l_fid = x.new_zeros(())

    l_spec = (
        multiscale_stft_loss(x, x_hat, n_ffts=spec_n_ffts)
        if gamma > 0 else x.new_zeros(())
    )

    total = l_rec + alpha * loss_vq + beta * l_fid + gamma * l_spec

    return {
        "loss":      total,
        "loss_rec":  l_rec,
        "loss_vq":   loss_vq,
        "loss_fid":  l_fid,
        "loss_spec": l_spec,
    }
