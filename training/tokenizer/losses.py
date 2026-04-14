"""
training/tokenizer/losses.py

L_total = L_rec + alpha * L_vq + beta * L_fiducial

fiducial loss: gradient-based (default) + optional point-wise weighted MSE
"""

import torch
import torch.nn.functional as F
from typing import Optional


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """L_rec: MSE between original and reconstructed beat."""
    return F.mse_loss(x_hat, x)


def gradient_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    L_fiducial (gradient proxy):
    ||∇x - ∇x̂||²   — QRS, P, T의 급격한 변화 구간 보존
    finite difference on time axis (last dim)
    """
    dx     = x[..., 1:] - x[..., :-1]
    dx_hat = x_hat[..., 1:] - x_hat[..., :-1]
    return F.mse_loss(dx_hat, dx)


def weighted_mse_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Point-wise weighted MSE.
    weights: same shape as x, higher near fiducial points.
    """
    if weights is None:
        return F.mse_loss(x_hat, x)
    loss = ((x_hat - x) ** 2 * weights).mean()
    return loss


def total_vqvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    loss_vq: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    use_gradient_loss: bool = True,
    fiducial_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Returns a dict with individual and total losses.
    """
    l_rec = reconstruction_loss(x, x_hat)

    if use_gradient_loss:
        l_fid = gradient_loss(x, x_hat)
    else:
        l_fid = weighted_mse_loss(x, x_hat, fiducial_weights)

    total = l_rec + alpha * loss_vq + beta * l_fid

    return {
        "loss":      total,
        "loss_rec":  l_rec,
        "loss_vq":   loss_vq,
        "loss_fid":  l_fid,
    }
