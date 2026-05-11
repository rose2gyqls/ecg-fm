import torch
import torch.nn as nn
from .encoder  import BeatEncoder
from .codebook import VQCodebook
from .decoder  import BeatDecoder

class VQVAE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        enc_cfg  = dict(cfg["encoder"])
        cb_cfg   = dict(cfg["codebook"])
        dec_cfg  = dict(cfg.get("decoder", {}))
        latent   = cfg.get("latent_dim", 256)
        enc_cfg.setdefault("latent_dim",    latent)
        dec_cfg.setdefault("latent_dim",    latent)
        cb_cfg.setdefault("embedding_dim",  latent)
        self.encoder  = BeatEncoder(**enc_cfg)
        self.codebook = VQCodebook(**cb_cfg)
        self.decoder  = BeatDecoder(**dec_cfg)
    def forward(self, x: torch.Tensor):
        z_e               = self.encoder(x)
        z_q, idx, l_vq, ppl, neg_h = self.codebook(z_e)
        x_hat             = self.decoder(z_q)
        return x_hat, {
            "loss_vq":     l_vq,
            "perplexity":  ppl,
            "neg_entropy": neg_h,
            "indices":     idx,
        }
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        self.eval()
        z_e = self.encoder(x)
        z_q, idx, *_ = self.codebook(z_e)
        return z_q, idx
    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.codebook.embedding(indices)
        return self.decoder(z_q)
