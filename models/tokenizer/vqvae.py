"""
models/tokenizer/vqvae.py

Full VQ-VAE: encoder + codebook + decoder
"""

import torch
import torch.nn as nn
from .encoder  import BeatEncoder
from .codebook import VQCodebook
from .decoder  import BeatDecoder


class VQVAE(nn.Module):
    """
    Beat-level VQ-VAE.

    encode()  : beat -> (z_q, indices)   — inference / tokenizer 역할
    forward() : beat -> (x_hat, loss_dict) — training 역할
    """

    def __init__(self, cfg: dict):
        super().__init__()
        enc_cfg  = dict(cfg["encoder"])
        cb_cfg   = dict(cfg["codebook"])
        dec_cfg  = dict(cfg.get("decoder", {}))
        latent   = cfg.get("latent_dim", 256)

        # config에 명시된 차원이 있으면 우선, 없으면 latent fallback (중복 kwarg 회피)
        enc_cfg.setdefault("latent_dim",    latent)
        dec_cfg.setdefault("latent_dim",    latent)
        cb_cfg.setdefault("embedding_dim",  latent)

        self.encoder  = BeatEncoder(**enc_cfg)
        self.codebook = VQCodebook(**cb_cfg)
        self.decoder  = BeatDecoder(**dec_cfg)

    # ── training forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, 1, W)
        Returns:
            x_hat    : (B, 1, W)
            loss_dict: {"loss_vq": ..., "perplexity": ..., "indices": ...}
        """
        z_e               = self.encoder(x)
        z_q, idx, l_vq, ppl = self.codebook(z_e)
        x_hat             = self.decoder(z_q)

        return x_hat, {
            "loss_vq":    l_vq,
            "perplexity": ppl,
            "indices":    idx,
        }

    # ── inference only ───────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """beat -> (z_q, indices)  — Phase 3 pretrain 입력 생성용"""
        self.eval()
        z_e = self.encoder(x)
        z_q, idx, _, _ = self.codebook(z_e)
        return z_q, idx

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices -> reconstructed beat"""
        z_q = self.codebook.embedding(indices)
        return self.decoder(z_q)
