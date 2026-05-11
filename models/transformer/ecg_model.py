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
    def __init__(self, cfg: dict):
        super().__init__()
        d  = cfg["d_model"]
        ctx = cfg.get("context", {})
        self.morph_emb = MorphologyEmbedding(cfg["codebook_size"], d)
        self.lead_emb  = LeadEmbedding(cfg.get("n_leads", 12), d)
        self.pos_emb   = BeatPositionEmbedding(cfg.get("max_beats", 20), d)
        self.rhythm_mlp = RhythmMLP(
            input_dim=ctx.get("rhythm_input_dim", 3),
            hidden=ctx.get("rhythm_hidden", 128),
            d_model=d,
            norm_mean=ctx.get("rhythm_mean"),
            norm_std=ctx.get("rhythm_std"),
        )
        self.global_ctx = GlobalContextCNN(
            in_channels=cfg.get("n_leads", 12),
            channels=ctx.get("stft_channels", [16, 32, 64]),
            d_model=d,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["num_layers"],
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)
        self.d_model = d
    def forward(
        self,
        indices:   torch.Tensor,
        rr_feats:  torch.Tensor,
        stft_map:  torch.Tensor,
        lead_ids:  Optional[torch.Tensor] = None,
        pad_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, L = indices.shape
        if lead_ids is None:
            lead_ids = torch.arange(L, device=indices.device)
        beat_pos = torch.arange(N, device=indices.device)
        lead_ids_exp = lead_ids.unsqueeze(0).expand(N, -1)
        beat_pos_exp = beat_pos.unsqueeze(1).expand(-1, L)
        morph = self.morph_emb(indices)
        rr_flat = rr_feats.view(B * N * L, 3)
        rhythm  = self.rhythm_mlp(rr_flat).view(B, N, L, -1)
        l_emb = self.lead_emb(lead_ids_exp)
        p_emb = self.pos_emb(beat_pos_exp)
        tokens = morph + rhythm + l_emb.unsqueeze(0) + p_emb.unsqueeze(0)
        tokens = tokens.view(B, N * L, self.d_model)
        g = self.global_ctx(stft_map).unsqueeze(1)
        seq = torch.cat([g, tokens], dim=1)
        out = self.transformer(seq, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return out
    def get_cls_repr(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        return out[:, 0, :]
