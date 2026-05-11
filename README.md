# ECG Foundation Model — Beat-level VQ-VAE + Masked Beat Modeling + Contrastive

12-lead ECG에서 **beat 단위 VQ-VAE 토크나이저**로 morphology를 이산 토큰으로 양자화하고, 그 위에 **Masked Beat Modeling + per-record SimCLR contrastive** Transformer foundation model을 학습한다. HEEDB(11.23M record) H5 데이터를 직접 streaming으로 읽어 DDP로 학습한다.

현재 제출용 최종 lineage는 **v4** — v3의 masked-beat pretraining objective는 유지하고, tokenizer 쪽 안정성 문제(record_mad outlier amplification, codebook EMA drift, dead-code restart shock)를 수정한 버전이다.

---

## Pipeline 개요

```
[Phase 1] Beat Tokenizer  (VQ-VAE, EMA cosine codebook)
   12-lead ECG (12, T)                     ← raw mV, gravity-quantized float16
       ↓ R-peak (Lead II + neurokit) + 검증
       ↓ extract beats (before=200ms, after=400ms)
       ↓ flat-beat 필터 (raw mV thresholds)
       ↓ ★ record_mad normalize: (sig − median) / (5·p75)
       ↓ resample → (M, 256), single-lead per beat
   BeatEncoder (shared 1D-CNN) → z_e → VQCodebook (cosine, EMA) → Decoder
   Loss = L_rec + α·L_vq + β·L_fid(grad) + γ·L_spec(multi-scale STFT)

[Phase 3] Masked Beat Modeling + SimCLR  (frozen tokenizer)
   record (12, T) → record_mad normalize → 30 beats × 12 leads tokenize
       ↓ apply_masking ×2 (independent RNG → two views)
   for each view:
       T_{i,j} = MorphEmb(z) + RhythmMLP(rr) + LeadEmb(j) + BeatPos(i)
       seq    = [g, T_{1,1}, ..., T_{N,12}],  g = STFT-CNN global token
       out    = Pre-LN Transformer Encoder(seq)        # (B, 1+N·12, d)
       cls    = out[:, 0, :]                            # patient-level summary
   Losses (averaged across two views):
       L_mlm = CE on beat-mask positions          (class-balanced weights)
       L_rr  = MSE  on rhythm-mask positions      (z-scored target)
       L_fid = MSE  on beat-mask positions        (Q-R, R-S z-scored target)
   Contrastive (cross-view):
       z_v = ProjMLP(cls_v) → L2-norm
       L_ctr = NT-Xent(z_1, z_2; τ=0.1) with all_gather DDP negative pool
   Total = w_mlm·L_mlm + w_rr·L_rr + w_fid·L_fid + w_ctr·L_ctr

[Phase 4] Fine-tuning  (benchmark/ repo)
   ECGFMHBEncoder reads model_cfg from ckpt → auto-picks normalize mode
   raw (12, T) → preprocess → indices + rr + stft → ECGFoundationModel
   out[:, 0, :] = CLS → ClassifierHead (linear / attention pooling)
```

---

## v4 핵심 변경

| 영역 | **v4** | 이유 |
|---|---|---|
| Tokenizer 정규화 | `record_mad_min_scale=0.05`, `record_mad_clip=8.0` | isoelectric lead가 scale floor에 걸리며 QRS가 과증폭되던 batch spike 차단. |
| Codebook EMA | `ema_decay=0.97`, `commitment_cost=0.25` | encoder drift를 EMA codebook이 늦게 따라가던 desync 완화. |
| Codebook usage | entropy regularizer `loss.delta=0.05` | V1~V6가 같은 dominant token으로 뭉치는 collapse 완화. |
| Dead-code restart | `restart_dead_every=2000`, `restart_dead_threshold=0.1` | restart shock 빈도와 강도 축소. |
| Tokenizer LR | `warmup_epochs=5` 실제 적용 | 초반 codebook/encoder 안착 안정화. |
| Best checkpoint | `best_window=5` smoothed val_loss | single validation outlier가 best.pt를 잠그는 문제 완화. |
| Pretrain objective | v3 objective 유지: MLM + RR + fiducial + SimCLR contrastive | tokenizer 개선 효과를 pretrain ablation에 단독 반영. |

제출용 실행 경로는 `*_v4.yaml`, `run_tokenizer_ablation_v4.sh`, `run_pretrain_ablation_v4.sh`, `training/pretrain/train.py` 기준이다.

---

## 환경 설정

```bash
conda env create -f environment.yaml
conda activate ecg-fm
```

기존 환경을 쓰는 경우에는 `python`/`torchrun`이 PATH에 있으면 된다. 특정 환경의 binary를 직접 지정하려면 `HBKIM_BIN=/path/to/env/bin`을 넘긴다.

> 필요한 경우: `pip install neurokit2 wfdb h5py scikit-learn tensorboard`

---

## 디렉토리 구조

```
ecg-fm/
├── configs/
│   ├── tokenizer/
│   │   └── vqvae_heedb_full_cb{128,256,512,1024,2048}_v4.yaml
│   ├── pretrain/
│   │   └── masked_beat_heedb_cb{128,256,512,1024,2048}_v4.yaml
│   └── finetune/
│       └── arrhythmia.yaml                                    # template; 실제 finetune은 benchmark/ 에서
│
├── data/
│   ├── preprocessing/
│   │   ├── heedb_io.py                # HEEDB H5 reader (lead 정렬)
│   │   ├── beat_segmentor.py          # neurokit R-peak + 검증 + RR + Q-R/R-S
│   │   ├── resampler.py               # ★ normalize_beat (zscore) + record_mad helpers
│   │   └── stft_extractor.py          # log-magnitude STFT (12, F, T')
│   └── datasets/
│       ├── heedb_beat_dataset.py      # Phase 1 (streaming, per-worker buffer)
│       ├── heedb_ecg_dataset.py       # Phase 3 (10초 ECG → beats+rr+stft)
│       ├── beat_dataset.py            # Phase 1 npy/h5 fallback
│       ├── ecg_dataset.py             # Phase 3 npy/h5 fallback
│       └── finetune_dataset.py        # ECGDataset + label
│
├── models/
│   ├── tokenizer/
│   │   ├── encoder.py                 # BeatEncoder: shared 1D-CNN, lead-blind
│   │   ├── codebook.py                # VQCodebook EMA + cosine + DDP all_reduce
│   │   ├── decoder.py                 # BeatDecoder: ConvTranspose mirror
│   │   └── vqvae.py                   # VQVAE wrapper
│   ├── context/
│   │   └── embeddings.py              # Morph/Lead/BeatPos/Rhythm/GlobalContextCNN
│   ├── transformer/
│   │   └── ecg_model.py               # ECGFoundationModel (Pre-LN Transformer)
│   └── heads/
│       ├── mlm_head.py                # MaskedBeatModelingHead / RR / Fiducial / Classifier
│       └── contrastive_head.py        # ★ ProjectionHead + nt_xent_loss (DDP-aware)
│
├── training/
│   ├── tokenizer/
│   │   ├── train.py                   # Phase 1 DDP loop + auto-resume + TB
│   │   └── losses.py                  # L_rec + α·L_vq + β·L_fid(grad) + γ·L_spec
│   ├── pretrain/
│   │   ├── train.py                   # ★ Phase 3 — two-view + contrastive + mask warmup
│   │   └── masking.py                 # ★ apply_masking + lead_dropout/mask_ratio schedules
│   └── finetune/
│       └── train.py                   # Phase 4 base (실제 finetune 평가는 benchmark/)
│
├── utils/                             # checkpointing / metrics / logging
│
├── scripts/
│   ├── run_tokenizer.sh               # Phase 1 launcher (단일 config)
│   ├── run_pretrain.sh                # Phase 3 launcher
│   ├── run_finetune.sh                # Phase 4 launcher (template)
│   ├── run_tokenizer_ablation_v4.sh   # v4 tokenizer codebook sweep
│   ├── run_pretrain_ablation_v4.sh    # v4 pretrain codebook sweep
│   └── build_full_heedb_filelist.py   # HEEDB → train/val 파일 리스트 (seed=42)
│
├── file_lists/                        # train/val .txt
├── checkpoints/                       # (.gitignore)
└── logs/                              # (.gitignore)
```

---

## 데이터 — HEEDB

- **루트**: HEEDB H5 파일이 있는 로컬 경로 (`--heedb-root` 또는 `HEEDB_ROOT`)
- **포맷**: `ECG/segments/0/signal` (12, T) float16, `metadata.attrs["fs"]`, `beat_annotation/sample` (있으면)
- **lead 순서**: `I, II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR` ([data/preprocessing/heedb_io.py:19](data/preprocessing/heedb_io.py#L19))
- **R-peak**: HEEDB 내장 annotation은 무시하고 **항상 Lead II + neurokit2** 로 재검출 ([heedb_beat_dataset.py:111](data/datasets/heedb_beat_dataset.py#L111))
- **target_fs=500Hz** (polyphase resample) → R-peak 기준 `before=200ms / after=400ms` (=300 samples) → `beat_length=256` 으로 resample
- **Normalize**: v4는 **record_mad** robust scaling을 사용하고, `record_mad_min_scale=0.05`, `record_mad_clip=8.0`으로 outlier amplification을 제한한다 ([resampler.py:48-86](data/preprocessing/resampler.py#L48-L86))
- **Noise/flat 필터**: raw mV 단위로 `ptp<0.1` / `std<0.01` beat drop (정규화 *전*에 적용)

### File list 생성

```bash
python scripts/build_full_heedb_filelist.py --heedb-root /path/to/heedb
# → file_lists/train_files_full.txt (~11.23M)  +  val_files_full.txt (10K holdout)
```

---

## 학습 실행 — v4 권장 흐름

모든 launcher는 `GPUS=0,1,2,3` 같은 env로 GPU를 지정할 수 있다. 모든 train 루프는 `ckpt_dir/last.pt` 가 있으면 **자동 resume**.

### ① Phase 1 — Tokenizer (5 codebook v4 ablation)

```bash
nohup ./scripts/run_tokenizer_ablation_v4.sh > tokenizer_v4.log 2>&1 &

# 단일 codebook만
ONLY=cb1024 ./scripts/run_tokenizer_ablation_v4.sh
```

`checkpoints/tokenizer_heedb_full_cb{128,256,512,1024,2048}_v4/best.pt` 산출. 모니터링: `loss_rec`, `loss_vq`, `perplexity`, `loss_spec`, `usage_entropy`.

### ② Phase 3 — Pretrain (5 codebook v4)

```bash
nohup ./scripts/run_pretrain_ablation_v4.sh > pretrain_v4.log 2>&1 &

# cb1024 단독으로 빠른 검증
ONLY=cb1024 ./scripts/run_pretrain_ablation_v4.sh
```

학습 step:
1. `beats (B,N,12,256)` → frozen tokenizer.encode → `indices (B,N,12)`
2. `apply_masking()` × **2 회 독립 호출** (서로 다른 RNG state) → view 1, view 2
3. 각 view: `ECGFoundationModel(masked, masked_rr, stft_masked)` → `out (B, 1+N·12, d)`
4. **MLM/RR/Fid loss**: 각 view에서 mask 위치만 모아 head로 → 두 view 평균
5. **Contrastive loss**: `ProjMLP(cls_v1)`, `ProjMLP(cls_v2)` → DDP all_gather → NT-Xent
6. `loss = w_mlm·L_mlm + w_rr·L_rr + w_fid·L_fid + w_ctr·L_ctr`

### ③ Phase 4 — Finetune

```bash
python -m training.finetune.train --config configs/finetune/arrhythmia.yaml
```

프로젝트 외부 benchmark repo를 쓰는 경우에는 v4 checkpoint 경로(`checkpoints/pretrain_heedb_cb*_v4/best.pt`)와 record_mad preprocessing을 맞춘다.

> Cache 주의: `record_mad`로 학습된 v4 모델은 `record_mad` 캐시가 필요. 기존 `zscore` 캐시를 재사용하면 manifest 불일치로 cache builder가 에러를 띄움. 새 cache_dir 지정 또는 `--normalize record_mad`로 빌드.

---

## 모델 사양 요약

| Stage | 파일 | 핵심 |
|---|---|---|
| Encoder (tokenizer) | [models/tokenizer/encoder.py](models/tokenizer/encoder.py) | shared 1D-CNN, channels [32,64,128,256], stride 2×4 → AdaptiveAvgPool → Linear(256). v4 uses l2-normalized latent vectors. |
| Codebook | [models/tokenizer/codebook.py](models/tokenizer/codebook.py) | EMA, cosine VQ. `_ema_update`에서 `dist.all_reduce(SUM)`로 모든 rank 동기화. Dead-code restart. |
| Decoder | [models/tokenizer/decoder.py](models/tokenizer/decoder.py) | Linear → reshape → ConvTranspose1d ×4 (encoder mirror). |
| Tokenizer Loss | [training/tokenizer/losses.py](training/tokenizer/losses.py) | `L_rec + α·L_vq + β·L_fid(grad) + γ·L_spec(multi-scale STFT)`. |
| Embeddings | [models/context/embeddings.py](models/context/embeddings.py) | MorphEmb(K+1, +1=MASK), LeadEmb(12), BeatPosEmb(30), RhythmMLP(3→256→d, z-score 내장), GlobalContextCNN. |
| Transformer | [models/transformer/ecg_model.py](models/transformer/ecg_model.py) | Pre-LN `nn.TransformerEncoder`. v4: 12 layers, d=512, FFN=2048, max_seq_len=384. |
| Masking | [training/pretrain/masking.py](training/pretrain/masking.py) | lead_dropout(prob, min_leads, schedule) → mask_beat_tokens_span(ratio, span) → mask_rhythm_features. cross_lead_aligned=True (lead redundancy 차단). |
| Heads | [models/heads/mlm_head.py](models/heads/mlm_head.py) | MaskedBeatModelingHead(d→K) / RR(d→3) / Fiducial(d→2) / Classifier. |
| ★ Contrastive | [models/heads/contrastive_head.py](models/heads/contrastive_head.py) | ProjectionHead(d→hidden→128) + L2-norm + nt_xent_loss(τ=0.1, DDP all_gather). 학습 후 discard. |

---

## 주요 하이퍼파라미터 (v4 main)

| 항목 | 값 | 위치 |
|---|---|---|
| target_fs | 500 Hz | 모든 v4 config |
| beat_length | 256 (300 raw → 256 resample) | |
| before/after_ms | 200 / 400 | |
| codebook K | {128, 256, 512, 1024, 2048} ablation, **1024 메인 예시** | |
| latent_dim | 256 | |
| Tokenizer normalize | **record_mad** + min_scale 0.05 + clip 8.0 | resampler.py |
| Tokenizer batch / lr / epochs | 1024/GPU · 3e-4 · 100 | |
| Transformer d_model / heads / layers | 512 / 8 / **12** | |
| max_beats_per_lead | 30 | seq_len 1+30·12=361 |
| mask_strategy / span_length | span / **3** | |
| beat / rhythm mask_ratio (max) | **0.5 / 0.5** | |
| mask_warmup | 0.15 → 0.5, **50 epochs linear** | |
| lead_dropout / min_leads | **0.15 / 10** | |
| Pretrain batch / lr / epochs | 64/GPU · 3e-4 · 200 | default 4-GPU launcher |
| early_stop_metric | **val_acc_nontop** | |
| Contrastive weight / τ / warmup | **0.3 / 0.1 / 5 ep** | NT-Xent on cls pair |
| Loss weights | morph=1.0, rhythm=1.0, fid=1.0, ctr=0.3 | class-balanced α=0.5 |

---

## Contrastive Learning — Positive / Negative 정의

**View 두 개**: 같은 record에 `apply_masking()`을 두 번 독립 호출. RNG state가 다르므로:
- view 1: lead {2, 5, 9} 가림 + beat span [3:6, 12:15] 가림
- view 2: lead {1, 7, 8} 가림 + beat span [5:8, 18:21] 가림

같은 환자의 같은 ECG지만 **누락된 정보가 다름** → encoder가 양쪽에서 비슷한 CLS를 만들려면 환자 고유 morphology + rhythm pattern을 잡아야 함.

**Positive / Negative**:
- DDP world W, per-rank batch B → 글로벌 N = W·B records
- `all_gather` 후 stack `Z = [z₁_all, z₂_all]` → shape (2N, 128)
- Anchor `i` (∈ [0, 2N)):
  - **Positive (1개)**: `(i + N) mod 2N` — 자기 다른 view
  - **Negative (2N − 2개)**: 같은 batch의 모든 다른 record (양 view 모두). Self는 `−∞`로 mask.
- Loss: `L_i = −log(exp(sim(i, pos)/τ) / Σ_{j≠i} exp(sim(i, j)/τ))`
- DDP gradient: local rank의 z만 grad 통과, remote rank z는 detach (negative pool로만 사용)

**진단 metric** (pretrain log + TB):
- `ctr_pos_sim`: positive cos sim 평균. 학습되면 0 → 0.7~0.9
- `ctr_neg_sim`: negative cos sim 평균. ~0 근처 유지
- `ctr_acc`: anchor가 자기 positive를 top-1으로 맞히는 비율. 0 → 1
- 초기 `ctr_acc < 0.1` 지속 → view augmentation 너무 강함 → mask_ratio / lead_dropout 낮춰야 함

---

## DDP 운용 노트

- **DDP launch**: `torchrun --standalone --nproc_per_node=$NPROC -m training.<phase>.train --config ...`
- **find_unused_parameters=True**: VQ codebook EMA + lead-dropout으로 일부 head가 unused step에 발생. proj_head도 contrastive_weight=0일 때 unused.
- **EMA codebook 동기화**: `_ema_update` 안에서 `dist.all_reduce(SUM)`. `broadcast_buffers=False`로 충분.
- **OMP_NUM_THREADS**: BLAS oversubscribe 방지로 1~4 강제 (launcher가 export).
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** — large batch fragmentation 완화.
- **자동 resume**: `ckpt_dir/last.pt` 자동 로드 (epoch/optimizer/scheduler/global_step/best_metric/proj_head 모두 복원).
- **Two-view memory**: v4 contrastive enabled 시 transformer activation 약 **2×**. d=512 12-layer + B=64 기준 OOM이 나면 batch를 절반으로 낮춘다.

---

## 추론 / 사용 예시

### Tokenizer로 beat → indices

```python
import torch, numpy as np
from models.tokenizer.vqvae import VQVAE
from data.preprocessing.resampler import compute_record_norm_stats, apply_record_norm

ck = torch.load("checkpoints/tokenizer_heedb_full_cb1024_v4/best.pt", map_location="cpu")
tok = VQVAE(ck["model_cfg"]); tok.load_state_dict(ck["model"]); tok.eval()

# v4 record_mad: per-record stat 한 번 계산하고 모든 beat에 적용
sig = np.random.randn(12, 5000).astype(np.float32) * 0.05   # raw mV
med, sc = compute_record_norm_stats(sig, min_scale=0.05)
sig_norm = apply_record_norm(sig, med, sc, clip=8.0)
# (extract / resample beats from sig_norm) → beats (M, 1, 256)
# z_q, indices = tok.encode(beats)
```

### Pretrained encoder via benchmark adapter

```python
from src.encoders.ecg_fm_hb import ECGFMHBEncoder
enc = ECGFMHBEncoder(checkpoint="checkpoints/pretrain_heedb_cb1024_v4/best.pt")
# enc.normalize 자동 = "record_mad" (model_cfg에 저장됨)
# enc(x) → (seq_feat, cls_pooled)
```

---

## 제출용 체크리스트

| 항목 | 기준 |
|---|---|
| Tokenizer configs | `configs/tokenizer/vqvae_heedb_full_cb*_v4.yaml` |
| Pretrain configs | `configs/pretrain/masked_beat_heedb_cb*_v4.yaml` |
| Tokenizer launcher | `scripts/run_tokenizer_ablation_v4.sh` |
| Pretrain launcher | `scripts/run_pretrain_ablation_v4.sh` |
| Single-run defaults | `run_tokenizer.sh`, `run_pretrain.sh` 모두 cb1024 v4 config |
| Main model | `models/transformer/ecg_model.py` |
