import torch
import torch.nn as nn

class LeadEmbedding(nn.Embedding):
    def __init__(self, n_leads: int = 12, d_model: int = 256):
        super().__init__(n_leads, d_model)

class BeatPositionEmbedding(nn.Embedding):
    def __init__(self, max_beats: int = 20, d_model: int = 256):
        super().__init__(max_beats, d_model)

class RhythmMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden: int = 128,
        d_model: int = 256,
        norm_mean: list | None = None,
        norm_std: list | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        if norm_mean is not None and norm_std is not None:
            mean_t = torch.tensor(norm_mean, dtype=torch.float32)
            std_t = torch.tensor(norm_std, dtype=torch.float32)
            if mean_t.numel() != input_dim or std_t.numel() != input_dim:
                raise ValueError(
                    f"RhythmMLP norm_{{mean,std}} length must match input_dim={input_dim}"
                )
            self.register_buffer("norm_mean", mean_t, persistent=True)
            self.register_buffer("norm_std", std_t, persistent=True)
            self._normalize = True
        else:
            self._normalize = False
    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            rr = (rr - self.norm_mean) / (self.norm_std + 1e-8)
        return self.net(rr)

class GlobalContextCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        channels: tuple = (16, 32, 64),
        d_model: int = 256,
    ):
        super().__init__()
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(in_ch, d_model)
    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        h = self.cnn(stft)
        h = self.pool(h).flatten(1)
        return self.project(h)

class MorphologyEmbedding(nn.Embedding):
    def __init__(self, codebook_size: int = 512, d_model: int = 256):
        super().__init__(codebook_size + 1, d_model)
        self.mask_token_id = codebook_size
    def get_mask_token(self, shape, device):
        return torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
