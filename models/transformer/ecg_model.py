"""
models/transformer/ecg_model.py

ECG Foundation Model.
Stage 2 Context Injection + Stage 3 Transformer Encoder를 통합한 메인 모델.

Token construction per beat (i=beat, j=lead):
    T_{i,j} = Emb(z_{i,j}) + RhythmMLP(p_{i,j}) + LeadEmb(j) + PosEmb(i)

Sequence:  [g, T_{1,1}, T_{1,2}, ..., T_{N,12}]
"""

import torch
import torch.nn as nn
from typing import Optional

from models.context.embeddings import (
    MorphologyEmbedding,
    LeadEmbedding,
    BeatPositionEmbedding,
    RhythmMLP,
    GlobalContextCNN,
)


class ECGFoundationModel(nn.Module):
    """
    Args (from config):
        d_model        : Transformer hidden dim
        nhead          : attention heads
        num_layers     : Transformer encoder layers
        dim_feedforward: FFN width
        dropout        : dropout
        codebook_size  : VQ-VAE codebook K
        n_leads        : 12
        max_beats      : maximum beats per lead (default 15)
        context.*      : rhythm/global context hyperparams
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d  = cfg["d_model"]
        ctx = cfg.get("context", {})

        # ── Token embeddings ────────────────────────────────────────────────
        self.morph_emb = MorphologyEmbedding(cfg["codebook_size"], d)
        self.lead_emb  = LeadEmbedding(cfg.get("n_leads", 12), d)
        self.pos_emb   = BeatPositionEmbedding(cfg.get("max_beats", 20), d)
        self.rhythm_mlp = RhythmMLP(
            input_dim=ctx.get("rhythm_input_dim", 3),
            hidden=ctx.get("rhythm_hidden", 128),
            d_model=d,
        )

        # ── Global context ───────────────────────────────────────────────────
        self.global_ctx = GlobalContextCNN(
            in_channels=cfg.get("n_leads", 12),
            channels=ctx.get("stft_channels", [16, 32, 64]),
            d_model=d,
        )
        # Learnable [CLS] token that will be replaced by g
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))

        # ── Transformer Encoder ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,          # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["num_layers"],
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)

        self.d_model = d

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        indices:   torch.Tensor,         # (B, N_beats, 12)  VQ indices
        rr_feats:  torch.Tensor,         # (B, N_beats, 12, 3) RR features
        stft_map:  torch.Tensor,         # (B, 12, F, T')
        lead_ids:  Optional[torch.Tensor] = None,  # (12,) or None
        pad_mask:  Optional[torch.Tensor] = None,  # (B, N_beats*12+1) bool True=pad
    ) -> torch.Tensor:
        """
        Returns:
            out : (B, 1 + N_beats*12, d_model)
                  out[:, 0, :] is the [CLS]/global representation
        """
        B, N, L = indices.shape          # L == n_leads == 12

        # ── Lead & position index tensors ────────────────────────────────────
        if lead_ids is None:
            lead_ids = torch.arange(L, device=indices.device)   # (12,)

        beat_pos = torch.arange(N, device=indices.device)       # (N,)
        lead_ids_exp = lead_ids.unsqueeze(0).expand(N, -1)      # (N, 12)
        beat_pos_exp = beat_pos.unsqueeze(1).expand(-1, L)      # (N, 12)

        # ── Token construction: T_{i,j} ─────────────────────────────────────
        # morphology
        morph = self.morph_emb(indices)      # (B, N, 12, d)

        # rhythm
        rr_flat = rr_feats.view(B * N * L, 3)
        rhythm  = self.rhythm_mlp(rr_flat).view(B, N, L, -1)   # (B,N,12,d)

        # lead / position
        l_emb = self.lead_emb(lead_ids_exp)           # (N, 12, d)
        p_emb = self.pos_emb(beat_pos_exp)             # (N, 12, d)

        # element-wise sum
        tokens = morph + rhythm + l_emb.unsqueeze(0) + p_emb.unsqueeze(0)
        # (B, N, 12, d)

        # flatten to sequence: (B, N*12, d)
        tokens = tokens.view(B, N * L, self.d_model)

        # ── Global context token g ───────────────────────────────────────────
        g = self.global_ctx(stft_map).unsqueeze(1)   # (B, 1, d)

        # prepend g as [CLS] replacement
        seq = torch.cat([g, tokens], dim=1)          # (B, 1+N*12, d)

        # ── Transformer ─────────────────────────────────────────────────────
        out = self.transformer(seq, src_key_padding_mask=pad_mask)
        out = self.norm(out)

        return out                                    # (B, 1+N*12, d)

    # ── convenience ─────────────────────────────────────────────────────────

    def get_cls_repr(self, *args, **kwargs) -> torch.Tensor:
        """(B, d_model) CLS token representation for downstream tasks."""
        out = self.forward(*args, **kwargs)
        return out[:, 0, :]
