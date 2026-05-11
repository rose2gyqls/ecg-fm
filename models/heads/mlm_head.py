import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBeatModelingHead(nn.Module):
    def __init__(self, d_model: int = 256, codebook_size: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, codebook_size),
        )
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

class MaskedRhythmHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

class MaskedFiducialHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

class ClassifierHead(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pooling: str = "cls",
    ):
        super().__init__()
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            rep = transformer_out[:, 0, :]
        else:
            rep = transformer_out[:, 1:, :].mean(1)
        return self.classifier(rep)
