"""
MoRyECG v6 — v5 + Global Refinement Block.

The only structural change vs v5 is one extra Transformer block at the very
end of the encoder stack:

    H, g  ──[ MoRyECG block × L_factor ]──►  H', g'
                                              │
                                              ▼
                                       [ GlobalRefineBlock ]
                                              │
                                              ▼
                                          H'', g''

The refinement block runs ONE round of FULL self-attention over the flat
sequence  [g, H_flat]  (length 1 + N·L = 361 for N=30, L=12), then a FFN.
Compared to the factorized blocks, this restores arbitrary token-to-token
attention paths that were severed by intra/inter-beat factorization.

Why: v5's downstream attention_probe lost ~1.1pp vs v4 because per-token
features had limited global context (info had to traverse multiple factor
hops). One late full-attention layer restores per-token global awareness
at minimal compute cost — only 361² attention pairs.

The factorized stack still does the heavy lifting (cheap per-layer cost,
useful inductive bias for ECG morphology). The refine block is the global
patch.

Loss / training contract is identical to v5 — same heads, same masking
pipeline. Only the model architecture and arch tag ("moryecg_v6") differ.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from models.transformer.moryecg_model import (
    MoRyECGFoundationModel,
    AdditiveBiasMHA,
)


# ──────────────────────────────────────────────────────────────────────────────
# Final full self-attention block
# ──────────────────────────────────────────────────────────────────────────────
class GlobalRefineBlock(nn.Module):
    """One Pre-LN Transformer block over the flat sequence [g, H_flat].

    Args:
        dim: model dim D
        num_heads: attention heads
        dim_feedforward: FFN width
        dropout: attention + FFN dropout

    forward:
        H : (B, N, L, D)
        g : (B, D)
        beat_valid_mask : (B, N) bool, optional. When provided, padded beat
            positions are excluded from attention via key_padding_mask. The
            [GLOB] slot (position 0) is always valid.

    Returns:
        H' : (B, N, L, D)
        g' : (B, D)
    """

    def __init__(self, dim: int, num_heads: int, dim_feedforward: int,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.ln_attn = nn.LayerNorm(self.dim)
        self.attn = AdditiveBiasMHA(self.dim, self.num_heads, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        H: torch.Tensor,                                 # (B, N, L, D)
        g: torch.Tensor,                                 # (B, D)
        beat_valid_mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, L, D = H.shape
        assert g.shape == (B, D), f"g shape {tuple(g.shape)} != ({B}, {D})"

        # ── Build flat sequence: [g, H_flat] of length 1 + N·L ─────────────
        H_flat = H.reshape(B, N * L, D)
        seq = torch.cat([g[:, None, :], H_flat], dim=1)        # (B, S, D)
        S = N * L + 1

        # ── Optional padding mask: excludes padded-beat positions ──────────
        # Each (beat_i, lead_j) is valid iff beat_i is a real (non-padded)
        # beat. The [GLOB] slot (index 0) is always valid.
        if beat_valid_mask is not None:
            glob_valid = torch.ones(B, 1, dtype=torch.bool, device=H.device)
            beat_valid_HL = (
                beat_valid_mask[:, :, None]              # (B, N, 1)
                .expand(-1, -1, L)                       # (B, N, L)
                .reshape(B, N * L)                       # (B, N·L)
            )
            valid = torch.cat([glob_valid, beat_valid_HL], dim=1)   # (B, S)
            key_padding_mask = ~valid                                # True = pad
        else:
            key_padding_mask = None

        # ── Attention + residual ───────────────────────────────────────────
        seq_norm = self.ln_attn(seq)
        seq = seq + self.attn(seq_norm, attn_bias=None,
                              key_padding_mask=key_padding_mask)

        # ── FFN + residual ─────────────────────────────────────────────────
        seq = seq + self.ffn(self.ln_ffn(seq))

        # ── Split back ─────────────────────────────────────────────────────
        g_out = seq[:, 0, :]                                       # (B, D)
        H_out = seq[:, 1:, :].reshape(B, N, L, D)                  # (B, N, L, D)

        # Zero padded beat slots so their representations don't drift
        # downstream — matches the contract enforced by every MoRyECG block.
        if beat_valid_mask is not None:
            H_out = H_out * beat_valid_mask[:, :, None, None].to(H_out.dtype)

        return H_out, g_out


# ──────────────────────────────────────────────────────────────────────────────
# v6 Foundation Model
# ──────────────────────────────────────────────────────────────────────────────
class MoRyECGv6FoundationModel(MoRyECGFoundationModel):
    """v6 = v5 + final GlobalRefineBlock.

    Adds one full-attention block after the factorized stack. The refine
    block runs BEFORE the final norm_h / norm_g (which we re-apply after
    refinement).

    Config additions vs v5:
        global_refine.enabled: bool, default True for v6
        global_refine.dim_feedforward: int, default = dim_feedforward (same
            as factorized blocks). Optional — set smaller to reduce param
            count.
        global_refine.dropout: float, default = dropout (same as blocks).
    """

    arch = "moryecg_v6"

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        gr_cfg = (cfg.get("global_refine") or {}) if cfg else {}
        self.global_refine_enabled = bool(gr_cfg.get("enabled", True))
        if self.global_refine_enabled:
            d_ff = int(gr_cfg.get("dim_feedforward", int(cfg["dim_feedforward"])))
            drop = float(gr_cfg.get("dropout", float(cfg.get("dropout", 0.1))))
            self.refine = GlobalRefineBlock(
                dim=self.d_model,
                num_heads=self.num_heads,
                dim_feedforward=d_ff,
                dropout=drop,
            )
            # Re-instantiate norm_h/norm_g so they sit AFTER the refine block.
            # (The base class registers norm_h/norm_g already; we keep them
            # but they will be applied after refinement, not before.)
        else:
            self.refine = None

    def forward(
        self,
        indices: torch.Tensor,
        rr_feats: torch.Tensor,
        stft: torch.Tensor,
        beat_valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if indices.dim() != 3:
            raise ValueError(f"indices expected (B, N, L); got {tuple(indices.shape)}")
        B, N, L = indices.shape

        # Same embedding + factorized stack as v5
        H = self._embed_tokens(indices, rr_feats)
        g = self._build_glob(stft)
        rr_bias_full = self._build_rr_bias(rr_feats)

        for block in self.blocks:
            H, g = block(H, g, rr_bias_full=rr_bias_full,
                         beat_valid_mask=beat_valid_mask)

        # ── v6 addition: one full-attention pass over [g, H_flat] ──────────
        if self.refine is not None:
            H, g = self.refine(H, g, beat_valid_mask=beat_valid_mask)

        # Final norms (applied AFTER the refine block in v6)
        H = self.norm_h(H)
        g = self.norm_g(g)
        # Final pad-zero (matches v5's fix): trained norm_h.bias would
        # otherwise leak into padded slots and pollute attention pooling.
        if beat_valid_mask is not None:
            H = H * beat_valid_mask[:, :, None, None].to(H.dtype)
        return {"H": H, "g": g}
