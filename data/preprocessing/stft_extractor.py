import numpy as np
import torch

def compute_stft_map(
    ecg: np.ndarray,
    fs: int = 500,
    n_fft: int = 256,
    hop_length: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    import torch
    window = torch.hann_window(n_fft)
    x = torch.from_numpy(ecg).float()
    specs = []
    for lead in range(x.shape[0]):
        s = torch.stft(
            x[lead],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag = s.abs().clamp(min=1e-8).log()
        specs.append(mag.numpy())
    stft_map = np.stack(specs, axis=0).astype(np.float32)
    if normalize:
        stft_map = (stft_map - stft_map.mean()) / (stft_map.std() + 1e-8)
    return stft_map
