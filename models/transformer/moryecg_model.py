"""
MoRyECG (Morphology + Rhythm ECG) Foundation Model — v5 transformer.

Replaces the v4 flat-sequence Transformer with a physiology-factorized block:

    Beat–lead tokens   H ∈ R^{B × N × L × D}        (kept 4D, NOT flattened)
    STFT-conditioned   g ∈ R^{B × D}

    MoRyECG block × L_layers:
        1. Intra-beat cross-lead attention   (per beat, 12 leads attend each other)
                                             — [GLOB] does NOT participate
        2. Inter-beat rhythm attention       (per lead, [GLOB] prepended)
                                             — RR-aware additive bias (shared MLP,
                                                per-head slopes)
        3. Lead-wise [GLOB] mean aggregation → canonical g
        4. FFN  (separate FFN_g for the global token by default)

Pre-training heads (in train_v5.py, not here):
    MLM on H[beat_mask] → codebook logits
    Contrastive (SimCLR) on ProjectionHead(g)

Tokenizer is v4-frozen — this model only consumes (indices, rr_feats, stft).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from models.context.embeddings import (
    MorphologyEmbedding,
    LeadEmbedding,
    BeatPositionEmbedding,
    RhythmMLP,
    STFTGlobalEncoder,
    GlobalContextCNN,
)
from models.context.rr_bias import PairwiseRRBiasMLP, pad_rr_bias_for_glob


# ──────────────────────────────────────────────────────────────────────────────
# Custom MHA with additive bias + key padding mask
# ──────────────────────────────────────────────────────────────────────────────
class AdditiveBiasMHA(nn.Module):
    """Multi-head self-attention that supports a per-batch additive score bias.

    PyTorch's nn.MultiheadAttention can take an additive `attn_mask`, but the
    shape gymnastics for a per-batch, per-head bias of shape (B, H, S, S) are
    error-prone (they require flattening the heads into the batch). Writing
    our own keeps the contract obvious:

        attn = (Q K^T) / sqrt(d_h)        # (B, H, S, S)
        attn = attn + attn_bias           # broadcast-compatible
        attn = attn.masked_fill(key_pad)  # set to -inf at padded keys
        attn = softmax(attn)
        out  = attn @ V                   # (B, H, S, d_h)
        out  = out_proj(out)

    Args:
        dim: model dim (= D)
        num_heads: H (must divide D)
        dropout: attention dropout (applied after softmax)

    forward args:
        x:                (B, S, D)  pre-LayerNorm'd input
        attn_bias:        (B, H, S, S) or None
        key_padding_mask: (B, S) bool, True = exclude  (or None)

    Returns:
        (B, S, D) attention output (no residual added — caller adds it).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, H, S, d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]             # each (B, H, S, d_h)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, S, S)

        if attn_bias is not None:
            # Support both (B, H, S, S) and (1, H, S, S) shapes.
            attn = attn + attn_bias

        if key_padding_mask is not None:
            # key_padding_mask: (B, S) bool, True = pad. Expand to (B, 1, 1, S)
            # so it broadcasts across heads and query positions.
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                   # (B, H, S, d_h)
        out = out.transpose(1, 2).reshape(B, S, D)    # (B, S, D)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Single MoRyECG block
# ──────────────────────────────────────────────────────────────────────────────
class MoRyECGBlock(nn.Module):
    """One Transformer block in the MoRyECG factorization.

    See module docstring for the data flow. All sublayers are Pre-LN.

    Args:
        dim, num_heads, dim_feedforward, dropout: standard Transformer knobs
        ffn_g_share: if True, the [GLOB] token re-uses the beat-token FFN
            (saves ~D·dim_feedforward params per block at the cost of
            forcing g and H into the same nonlinearity). Default False.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        ffn_g_share: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.ffn_g_share = bool(ffn_g_share)

        # Pre-LN for both attentions and the FFN. Each stage gets its own LN
        # following the standard Transformer recipe.
        self.ln_lead = nn.LayerNorm(self.dim)
        self.lead_attn = AdditiveBiasMHA(self.dim, self.num_heads, dropout=dropout)

        self.ln_rhythm = nn.LayerNorm(self.dim)
        self.rhythm_attn = AdditiveBiasMHA(self.dim, self.num_heads, dropout=dropout)

        self.ln_ffn = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.dim),
            nn.Dropout(dropout),
        )

        if self.ffn_g_share:
            self.ln_g = nn.LayerNorm(self.dim)
            self.ffn_g = None
        else:
            self.ln_g = nn.LayerNorm(self.dim)
            self.ffn_g = nn.Sequential(
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
        rr_bias_full: Optional[torch.Tensor] = None,     # (B, H, N+1, N+1)
        beat_valid_mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, L, D = H.shape
        assert g.shape == (B, D), f"g shape {tuple(g.shape)} != ({B}, {D})"

        # ── 1. Intra-beat cross-lead attention ─────────────────────────────
        # Each "sequence" is the 12 leads of one beat; [GLOB] is intentionally
        # excluded so the attention head can specialize on cross-lead morphology
        # without being distracted by the recording-level context.
        # Reshape: (B, N, L, D) → (B*N, L, D)
        X = H.reshape(B * N, L, D)
        X_norm = self.ln_lead(X)
        # No bias, no key_padding_mask (all 12 leads always present in the
        # token tensor; lead_dropout zeros their content via MASK token but
        # does not change the slot count).
        X = X + self.lead_attn(X_norm, attn_bias=None, key_padding_mask=None)
        H = X.reshape(B, N, L, D)

        # ── 2. Inter-beat rhythm attention with [GLOB] ─────────────────────
        # Each "sequence" is the N beats of one lead, with [GLOB] prepended.
        # [GLOB] is replicated per lead for the attention pass and merged back
        # by mean aggregation after the residual.
        # Reshape: (B, N, L, D) → (B, L, N, D)
        X = H.permute(0, 2, 1, 3).contiguous()           # (B, L, N, D)
        g_rep = g[:, None, None, :].expand(B, L, 1, D)    # (B, L, 1, D)
        X = torch.cat([g_rep, X], dim=2)                  # (B, L, N+1, D)
        X = X.reshape(B * L, N + 1, D)                    # (B*L, N+1, D)
        X_norm = self.ln_rhythm(X)

        # RR bias: (B, H, N+1, N+1) → (B*L, H, N+1, N+1) by lead-replication.
        if rr_bias_full is not None:
            assert rr_bias_full.shape == (B, self.num_heads, N + 1, N + 1), (
                f"rr_bias_full shape {tuple(rr_bias_full.shape)} != "
                f"({B}, {self.num_heads}, {N+1}, {N+1})"
            )
            bias_repl = rr_bias_full[:, None, :, :, :]                    # (B, 1, H, N+1, N+1)
            bias_repl = bias_repl.expand(B, L, self.num_heads, N + 1, N + 1)
            bias_repl = bias_repl.reshape(B * L, self.num_heads, N + 1, N + 1)
        else:
            bias_repl = None

        # key_padding_mask: pad positions among the N beats are excluded;
        # [GLOB] (slot 0) is always valid. Then replicated per lead.
        if beat_valid_mask is not None:
            glob_valid = torch.ones(B, 1, dtype=torch.bool, device=H.device)
            valid = torch.cat([glob_valid, beat_valid_mask], dim=1)        # (B, N+1)
            kpm_repl = valid[:, None, :].expand(B, L, N + 1)
            kpm_repl = kpm_repl.reshape(B * L, N + 1)
            key_padding_mask = ~kpm_repl                                    # True = pad
        else:
            key_padding_mask = None

        X = X + self.rhythm_attn(
            X_norm, attn_bias=bias_repl, key_padding_mask=key_padding_mask
        )
        X = X.reshape(B, L, N + 1, D)

        # ── 3. Lead-wise [GLOB] aggregation ────────────────────────────────
        g_leads = X[:, :, 0, :]                                             # (B, L, D)
        g = g_leads.mean(dim=1)                                             # (B, D)

        # Recover H from non-[GLOB] positions and restore (B, N, L, D) layout.
        H = X[:, :, 1:, :].permute(0, 2, 1, 3).contiguous()                 # (B, N, L, D)

        # ── 4. FFN ─────────────────────────────────────────────────────────
        H = H + self.ffn(self.ln_ffn(H))
        if self.ffn_g_share:
            g = g + self.ffn(self.ln_g(g))
        else:
            g = g + self.ffn_g(self.ln_g(g))

        # Zero out padded beat slots so their representations don't drift.
        if beat_valid_mask is not None:
            H = H * beat_valid_mask[:, :, None, None].to(H.dtype)

        return H, g


# ──────────────────────────────────────────────────────────────────────────────
# Full MoRyECG Foundation Model
# ──────────────────────────────────────────────────────────────────────────────
class MoRyECGFoundationModel(nn.Module):
    """v5 ECG foundation model.

    Required cfg fields (matches v5 yaml schema):
        d_model, nhead, num_layers, dim_feedforward, dropout
        codebook_size, n_leads, max_beats
        context.rhythm_input_dim, rhythm_hidden, [rhythm_mean, rhythm_std]
        context.stft_channels (list[int]), stft_dropout (float)
        context.rr_bias.{enabled, hidden, init_zero}
        ffn_g_share (bool, default False)
        glob_aggregation: only "mean" supported in v5
    """

    arch = "moryecg"

    def __init__(self, cfg: dict):
        super().__init__()
        d         = int(cfg["d_model"])
        nhead     = int(cfg["nhead"])
        n_layers  = int(cfg["num_layers"])
        d_ff      = int(cfg["dim_feedforward"])
        dropout   = float(cfg.get("dropout", 0.1))
        n_leads   = int(cfg.get("n_leads", 12))
        max_beats = int(cfg.get("max_beats", 30))
        ctx       = cfg.get("context", {}) or {}
        ffn_share = bool(cfg.get("ffn_g_share", False))

        self.d_model   = d
        self.num_heads = nhead
        self.num_layers = n_layers
        self.n_leads = n_leads
        self.max_beats = max_beats

        # ── Token embeddings ────────────────────────────────────────────────
        self.morph_emb = MorphologyEmbedding(int(cfg["codebook_size"]), d)
        self.lead_emb  = LeadEmbedding(n_leads, d)
        self.pos_emb   = BeatPositionEmbedding(max_beats, d)
        self.rhythm_mlp = RhythmMLP(
            input_dim=int(ctx.get("rhythm_input_dim", 3)),
            hidden=int(ctx.get("rhythm_hidden", 256)),
            d_model=d,
            norm_mean=ctx.get("rhythm_mean"),
            norm_std=ctx.get("rhythm_std"),
        )

        # ── STFT-conditioned [GLOB] token ───────────────────────────────────
        # Slim encoder by default ([8, 16, 32]) plus optional STFT-bin dropout
        # — both knobs reduce the encoder's capacity to memorize per-beat
        # spectral signatures and complement the explicit STFT masking that
        # train_v5.py applies on the input.
        stft_channels = ctx.get("stft_channels", [8, 16, 32])
        stft_dropout = float(ctx.get("stft_dropout", 0.0))
        self.global_ctx = STFTGlobalEncoder(
            in_channels=n_leads,
            channels=tuple(stft_channels),
            d_model=d,
            input_dropout=stft_dropout,
        )
        # Learnable bias added to the projected STFT vector — gives the model
        # a fallback identity for [GLOB] when the STFT input is fully zeroed
        # (worst-case leakage prevention).
        self.glob_token = nn.Parameter(torch.zeros(1, d))
        nn.init.normal_(self.glob_token, mean=0.0, std=0.02)

        # ── Pairwise RR bias (shared across blocks) ─────────────────────────
        rr_bias_cfg = (ctx.get("rr_bias") or {})
        self.rr_bias_enabled = bool(rr_bias_cfg.get("enabled", True))
        if self.rr_bias_enabled:
            self.rr_bias_mlp = PairwiseRRBiasMLP(
                num_heads=nhead,
                hidden=int(rr_bias_cfg.get("hidden", 64)),
                init_zero=bool(rr_bias_cfg.get("init_zero", True)),
            )
        else:
            self.rr_bias_mlp = None

        # ── L stacked MoRyECG blocks ────────────────────────────────────────
        self.blocks = nn.ModuleList([
            MoRyECGBlock(
                dim=d,
                num_heads=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                ffn_g_share=ffn_share,
            ) for _ in range(n_layers)
        ])
        self.norm_h = nn.LayerNorm(d)
        self.norm_g = nn.LayerNorm(d)

        agg = str(cfg.get("glob_aggregation", "mean")).lower()
        if agg != "mean":
            raise NotImplementedError(
                f"v5 only supports glob_aggregation='mean', got '{agg}'"
            )

    # ── Embedding ───────────────────────────────────────────────────────────
    def _embed_tokens(self, indices: torch.Tensor, rr_feats: torch.Tensor) -> torch.Tensor:
        """(B, N, L) + (B, N, L, 3) → (B, N, L, D)."""
        B, N, L = indices.shape
        device = indices.device

        morph = self.morph_emb(indices)                                   # (B, N, L, D)

        # rr_feats is lead-invariant by construction (dataset replicates the
        # same triplet across leads). Compute the rhythm vector once per beat.
        rr_per_beat = rr_feats[:, :, 0, :]                                 # (B, N, 3)
        rhythm = self.rhythm_mlp(rr_per_beat.reshape(B * N, -1)).reshape(B, N, -1)
        rhythm = rhythm.unsqueeze(2)                                       # (B, N, 1, D)

        beat_pos = torch.arange(N, device=device)
        lead_ids = torch.arange(L, device=device)
        p_emb = self.pos_emb(beat_pos)[None, :, None, :]                   # (1, N, 1, D)
        l_emb = self.lead_emb(lead_ids)[None, None, :, :]                  # (1, 1, L, D)

        return morph + rhythm + p_emb + l_emb                              # (B, N, L, D)

    def _build_glob(self, stft: torch.Tensor) -> torch.Tensor:
        """(B, n_leads, F, T') → (B, D)."""
        return self.glob_token + self.global_ctx(stft)

    def _build_rr_bias(self, rr_feats: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.rr_bias_enabled:
            return None
        bias = self.rr_bias_mlp(rr_feats)                                   # (B, H, N, N)
        return pad_rr_bias_for_glob(bias)                                   # (B, H, N+1, N+1)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        indices: torch.Tensor,                              # (B, N, L) long
        rr_feats: torch.Tensor,                             # (B, N, L, 3) float
        stft: torch.Tensor,                                 # (B, n_leads, F, T') float
        beat_valid_mask: Optional[torch.Tensor] = None,     # (B, N) bool
    ) -> dict:
        """Returns {"H": (B, N, L, D), "g": (B, D)} ready for downstream heads."""
        if indices.dim() != 3:
            raise ValueError(f"indices expected (B, N, L); got {tuple(indices.shape)}")
        B, N, L = indices.shape

        H = self._embed_tokens(indices, rr_feats)             # (B, N, L, D)
        g = self._build_glob(stft)                             # (B, D)
        rr_bias_full = self._build_rr_bias(rr_feats)           # (B, H, N+1, N+1) or None

        for block in self.blocks:
            H, g = block(H, g, rr_bias_full=rr_bias_full,
                         beat_valid_mask=beat_valid_mask)

        H = self.norm_h(H)
        g = self.norm_g(g)
        # Re-apply the padding mask AFTER the final LayerNorm. norm_h has a
        # learned bias that, post-training, would otherwise turn padded zero
        # slots into non-zero (= bias) values in the output — which then
        # contaminate downstream attention pooling. No-op when the caller
        # didn't pass a mask.
        if beat_valid_mask is not None:
            H = H * beat_valid_mask[:, :, None, None].to(H.dtype)
        return {"H": H, "g": g}

    def forward_flat(
        self,
        indices: torch.Tensor,
        rr_feats: torch.Tensor,
        stft: torch.Tensor,
        beat_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adapter-compatible flat sequence layout: (B, 1 + N*L, D).

        out[:, 0]  == g       (CLS-style pooled representation)
        out[:, 1:] == H.view(B, N*L, D)   (per-token features)

        Used by benchmark/src/encoders/ecg_fm_hb.py so the existing
        finetune contract `(seq_feat, pooled) = out[:, 1:], out[:, 0]` works
        without changes.
        """
        out = self.forward(indices, rr_feats, stft, beat_valid_mask=beat_valid_mask)
        H, g = out["H"], out["g"]
        B, N, L, D = H.shape
        return torch.cat([g[:, None, :], H.reshape(B, N * L, D)], dim=1)
