# MoRyECG

This repository contains a beat-tokenized 12-lead ECG foundation model. The pipeline has two training stages: a beat-level VQ-VAE tokenizer and a Transformer encoder trained with masked beat modeling, rhythm reconstruction, fiducial reconstruction, and contrastive learning.

## Model Structure

1. **Beat tokenizer**
   - Input: raw 12-lead ECG records.
   - R-peaks are detected from Lead II.
   - Beats are extracted with a 200 ms pre-R and 400 ms post-R window.
   - Each beat is resampled to 256 samples.
   - A shared 1D convolutional encoder maps each single-lead beat to a latent vector.
   - A cosine EMA vector quantizer converts latent vectors to discrete morphology tokens.
   - A convolutional decoder reconstructs the beat waveform.

2. **Pretraining encoder**
   - The tokenizer is frozen.
   - Each record is converted to beat tokens, RR features, fiducial features, and an STFT context map.
   - Masking is applied to morphology tokens and rhythm features.
   - Token embeddings combine morphology, rhythm, lead identity, beat position, and global STFT context.
   - A Transformer encoder produces per-token features and a global summary token.
   - Training losses include morphology classification, RR regression, fiducial regression, and contrastive loss between independently masked views.

3. **Fine-tuning**
   - The pretrained encoder can be fine-tuned with a task-specific classifier head.
   - The included fine-tuning config is a template and expects a local labeled ECG dataset.

## Setup

```bash
conda env create -f environment.yaml
conda activate ecg-fm
```

If you already have an environment, make sure `python` and `torchrun` are available on `PATH`. The shell launchers also accept `MoRyECG_BIN=/path/to/env/bin` when you want to use a specific environment without activation.

## Prepare File Lists

```bash
python scripts/build_full_heedb_filelist.py --heedb-root /path/to/heedb
```

This writes:

```text
file_lists/train_files_full.txt
file_lists/val_files_full.txt
```

The pretraining configs use shorter file-list names by default:

```text
file_lists/train_files.txt
file_lists/val_files.txt
```

You can either create those files directly or copy/symlink them from the full lists.

## Train the Tokenizer

Run one tokenizer config directly:

```bash
python -m training.tokenizer.train --config configs/tokenizer/vqvae_heedb_full_cb1024_v4.yaml
```

Run the full codebook sweep:

```bash
ONLY=cb1024 ./scripts/run_tokenizer_ablation_v4.sh
```

Remove `ONLY=cb1024` to run all configured codebook sizes.

## Train the Pretraining Encoder

Run one pretraining config directly:

```bash
python -m training.pretrain.train --config configs/pretrain/masked_beat_heedb_cb1024_v4.yaml
```

Run the full codebook sweep:

```bash
ONLY=cb1024 ./scripts/run_pretrain_ablation_v4.sh
```

Remove `ONLY=cb1024` to run all configured codebook sizes.

## Fine-tune

```bash
python -m training.finetune.train --config configs/finetune/arrhythmia.yaml
```

Update the config with your labeled dataset path, number of classes, and pretrained checkpoint path before running.

## Outputs

Training writes checkpoints under `checkpoints/` and logs under `logs/`. Both directories are ignored by git.
