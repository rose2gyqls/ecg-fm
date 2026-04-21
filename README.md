# ECG Foundation Model — Beat-level VQ-VAE + Masked Beat Modeling

12-lead ECG에서 **beat 단위 VQ-VAE 토크나이저**로 morphology를 이산 토큰으로 양자화하고, 그 위에 **Masked Beat Modeling** 기반 Transformer foundation model을 학습하는 구현체. HEEDB(11.23M record) H5 데이터를 직접 streaming으로 읽어 7-GPU DDP로 학습한다.

---

## Pipeline 개요

```
[Phase 1] Beat Tokenizer  (VQ-VAE, EMA codebook)
   12-lead ECG → R-peak 검출(Lead II) → beat segment (300 samples)
                → resample 256 → Shared 1D-CNN Encoder
                → VQ Codebook(K) → Decoder → reconstruction

[Phase 3] Masked Beat Modeling  (frozen tokenizer)
   beats(B,N,12,256) → tokenizer.encode → indices(B,N,12)
   token T_{i,j} = MorphEmb(z) + RhythmMLP(rr) + LeadEmb(j) + BeatPos(i)
   sequence  [g, T_{1,1}, ..., T_{N,12}] (g = STFT-CNN global context)
   Transformer Encoder + Pre-LN → MLM head (token CE) + RR head (MSE)
   + lead dropout 정규화

[Phase 4] Fine-tuning
   pretrained Transformer + ClassifierHead (CLS pooling) → AUROC/F1/Acc
```

---

## 디렉토리 구조

```
ecg-fm/
├── configs/
│   ├── tokenizer/
│   │   ├── vqvae_base.yaml                   # 일반 npy 입력 템플릿
│   │   ├── vqvae_heedb.yaml                  # HEEDB 200K subset, 30 epoch
│   │   ├── vqvae_heedb_full_cb256.yaml       # HEEDB 전체, K=256, 100ep
│   │   ├── vqvae_heedb_full_cb512.yaml       # HEEDB 전체, K=512, 100ep ★
│   │   ├── vqvae_heedb_full_cb1024.yaml      # HEEDB 전체, K=1024
│   │   └── vqvae_heedb_full_cb2048.yaml      # HEEDB 전체, K=2048
│   ├── pretrain/
│   │   ├── masked_beat_base.yaml             # 일반 입력 템플릿
│   │   └── masked_beat_heedb.yaml            # HEEDB 200K subset, 100ep
│   └── finetune/
│       └── arrhythmia.yaml
│
├── data/
│   ├── preprocessing/
│   │   ├── heedb_io.py                       # HEEDB H5 reader (lead 정렬 포함)
│   │   ├── beat_segmentor.py                 # neurokit R-peak + 검증 + RR features
│   │   ├── resampler.py                      # polyphase signal resample + beat resample + zscore
│   │   └── stft_extractor.py                 # log-magnitude STFT (12, F, T')
│   └── datasets/
│       ├── heedb_beat_dataset.py             # Phase 1 (streaming, per-worker buffer)
│       ├── heedb_ecg_dataset.py              # Phase 3 (10초 ECG → beats+rr+stft)
│       ├── beat_dataset.py                   # Phase 1 npy/h5 fallback
│       ├── ecg_dataset.py                    # Phase 3 npy/h5 fallback
│       └── finetune_dataset.py               # ECGDataset + label
│
├── models/
│   ├── tokenizer/
│   │   ├── encoder.py                        # BeatEncoder: (B,1,256)→(B,256) shared CNN
│   │   ├── codebook.py                       # VQCodebook EMA (DDP all_reduce 동기)
│   │   ├── decoder.py                        # BeatDecoder: ConvTranspose mirror
│   │   └── vqvae.py                          # VQVAE wrapper + .encode() / .decode_indices()
│   ├── context/
│   │   └── embeddings.py                     # MorphologyEmb / LeadEmb / BeatPosEmb / RhythmMLP / GlobalContextCNN
│   ├── transformer/
│   │   └── ecg_model.py                      # ECGFoundationModel (Pre-LN Transformer Encoder)
│   └── heads/
│       └── mlm_head.py                       # MaskedBeatModelingHead / MaskedRhythmHead / ClassifierHead
│
├── training/
│   ├── tokenizer/
│   │   ├── train.py                          # Phase 1 DDP loop + auto-resume + TB
│   │   └── losses.py                         # L_rec + α·L_vq + β·L_fiducial(gradient)
│   ├── pretrain/
│   │   ├── train.py                          # Phase 3 DDP loop + auto-resume + TB
│   │   └── masking.py                        # beat / rhythm / lead-dropout 마스킹
│   └── finetune/
│       └── train.py                          # Phase 4 + sklearn AUROC/F1/Acc
│
├── utils/
│   ├── checkpointing.py                      # save/load (model_cfg 동봉)
│   ├── logging_utils.py                      # CSV MetricLogger / WandbLogger
│   └── metrics.py                            # codebook usage·perplexity, SNR/PRD, clf metrics
│
├── scripts/
│   ├── run_tokenizer.sh                      # Phase 1 launcher (DDP, GPUS env)
│   ├── run_pretrain.sh                       # Phase 3 launcher
│   ├── run_finetune.sh                       # Phase 4 launcher (task name → config)
│   ├── run_tokenizer_ablation.sh             # cb256/1024/2048 sequential ablation + stamps
│   ├── build_full_heedb_filelist.py          # HEEDB 전체 → train/val 분할 (seed=42)
│   ├── heedb_sanity.py                       # H5 파이프라인 한두 파일 검증
│   └── smoke_test_beat_pipeline.py           # R-peak 검증 + noise filter drop 통계
│
├── file_lists/
│   ├── train_files_full.txt   (11,229,230)   # 전체 HEEDB train
│   ├── val_files_full.txt     (10,000)        # 전체 HEEDB val (seed=42 holdout)
│   ├── train_files.txt        (190,000)       # 200K subset의 train
│   ├── val_files.txt          (10,000)
│   └── subset200k.txt         (200,000)
│
├── checkpoints/                              # (.gitignore)
│   ├── tokenizer_heedb/                      # 200K subset, K=512, 30ep
│   ├── tokenizer_heedb_full_cb512/           # 전체 HEEDB, K=512, 100ep ★현재 메인
│   ├── tokenizer_heedb_full_cb256/           # ablation 진행 중
│   └── pretrain_heedb/                       # masked beat modeling, 100ep
│
├── logs/                                     # (.gitignore)
│   ├── tokenizer_heedb_full_cb512/{metrics.csv, tb/}
│   ├── pretrain_heedb/{metrics.csv, tb/}
│   └── ablation_runs/{cb256_*.log, _stamps/}
│
├── notebooks/
│   ├── check_codebook.ipynb                  # codebook usage / 재구성 시각화
│   └── make_vqvae_figure.py                  # 논문/슬라이드용 figure 생성
│
├── environment.yaml                          # conda env "hbkim" (CUDA 12.1, torch≥2.1)
└── README.md
```

---

## 환경 설정

```bash
conda env create -f environment.yaml      # 또는 기존 hbkim env에 pip 추가
conda activate hbkim
pip install neurokit2 wfdb h5py scikit-learn tensorboard
```

이미 서버에 conda env가 있으면 `/home/irteam/local-node-d/_conda/envs/hbkim/bin` 의 python/torchrun을 직접 사용한다 (`scripts/run_tokenizer.sh` 의 `HBKIM_BIN` 변수).

---

## 데이터 — HEEDB

- **루트**: `/home/irteam/ddn-opendata1/h5/heedb`
- **포맷**: `ECG/segments/0/signal` (12, T) float16, `metadata.attrs["fs"]`(가변), `beat_annotation/sample` (선택)
- **lead 순서**: `I, II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR` ([data/preprocessing/heedb_io.py:19](data/preprocessing/heedb_io.py#L19))
- **R-peak**: HEEDB 내장 annotation은 무시하고 일관성을 위해 **항상 Lead II + neurokit2** 로 재검출 ([data/datasets/heedb_beat_dataset.py:111](data/datasets/heedb_beat_dataset.py#L111))
- **target_fs=500Hz** 로 polyphase resample → R-peak 기준 `before=200ms / after=400ms`(=300 samples) 추출 → `beat_length=256` 으로 resample → z-score
- **R-peak 검증**: 경계 거리 + 국소 극대점(±10 샘플) 검증 ([data/preprocessing/beat_segmentor.py:59-89](data/preprocessing/beat_segmentor.py#L59-L89))
- **Noise/flat beat 필터**: `ptp<0.1mV` 또는 `std<0.01mV` 인 beat drop

### File list 생성

```bash
python scripts/build_full_heedb_filelist.py
# → file_lists/train_files_full.txt (11,229,230)  +  val_files_full.txt (10,000)
```

### 파이프라인 sanity check

```bash
python -m scripts.heedb_sanity --data_dir /home/irteam/ddn-opendata1/h5/heedb --n 3
python scripts/smoke_test_beat_pipeline.py    # R-peak/noise filter drop 통계
```

### 일반 npy/h5 입력 (HEEDB 외)

```python
np.save("data_dir/train/record_001.npy", {
    "signal": ecg_array,       # (12, T) float32
    "rpeaks": rpeak_indices,   # (N,) int, 선택
    "label":  0,               # int, fine-tune 시
})
```
config의 `data.source` 를 생략(기본 `npy`) 하고 `data_dir` 를 지정하면 `BeatDataset` / `ECGDataset` 가 동작한다.

---

## 학습 실행

모든 launcher는 `GPUS=0,1,2,...` env로 GPU를 지정하고, `nproc_per_node` 를 자동 계산한다. 모든 train 루프는 `ckpt_dir/last.pt` 가 있으면 **자동 resume** 한다.

### Phase 1 — Beat Tokenizer

```bash
# 단일 config (DDP)
GPUS=0,1,2,3,4,5,6 bash scripts/run_tokenizer.sh \
    configs/tokenizer/vqvae_heedb_full_cb512.yaml

# resume 명시
RESUME=checkpoints/tokenizer_heedb_full_cb512/epoch_050.pt \
GPUS=0,1,2,3,4,5,6 bash scripts/run_tokenizer.sh \
    configs/tokenizer/vqvae_heedb_full_cb512.yaml
```

#### Codebook size ablation

`cb512` 는 이미 학습 완료. 나머지(256/1024/2048)를 순차 실행:

```bash
nohup ./scripts/run_tokenizer_ablation.sh > ablation.log 2>&1 &

# 옵션
ONLY=cb1024       ./scripts/run_tokenizer_ablation.sh
SKIP=cb256        ./scripts/run_tokenizer_ablation.sh
FORCE_RESTART=1   ./scripts/run_tokenizer_ablation.sh   # stamp 무시 재시작
```

각 run은 `logs/ablation_runs/{tag}_{ts}.log` 에 기록되고, 성공 시 `logs/ablation_runs/_stamps/{tag}.done` stamp가 남는다.

**모니터링 지표** (`logs/.../metrics.csv` + TensorBoard `tb/`):
- `loss_rec` / `loss_fid`(gradient) — 복원 품질
- `loss_vq` — commitment loss (EMA 모드)
- `perplexity` — 코드북 사용 entropy의 exp. 높을수록 균등.
  - 진단: `usage<10%` 또는 `perplexity<K/10` → collapse 의심 → EMA decay 낮춤(0.99→0.95) 또는 lr 점검

### Phase 3 — Masked Beat Modeling

```bash
GPUS=0,1,2,3,4,5,6 bash scripts/run_pretrain.sh \
    configs/pretrain/masked_beat_heedb.yaml
```

Launcher가 `tokenizer.ckpt` 존재를 사전 확인. 학습 step:

1. `beats (B,N,12,256)` → frozen `tokenizer.encode` → `indices (B,N,12)`
2. `apply_masking()` — lead dropout → beat MASK → rhythm MASK 순서
3. `ECGFoundationModel(masked_indices, masked_rr, stft)` → `(B, 1+N·12, d)`
4. masked 위치만 모아 MLM head (cross-entropy over codebook), RR head (MSE)
5. `loss = morphology_weight · L_mlm + rhythm_weight · L_rr`

### Phase 4 — Fine-tuning

```bash
bash scripts/run_finetune.sh arrhythmia
bash scripts/run_finetune.sh configs/finetune/custom.yaml   # 직접 경로
```

`base_pretrain_ckpt` 의 `model_cfg` 가 함께 저장되어 있어 모델 재구성 자동. `model.classifier.freeze_layers=N` 또는 `freeze_transformer: true` 로 부분 freeze 가능. AUROC/F1/Acc는 sklearn 으로 계산 (`training/finetune/train.py:53-84`).

---

## 모델 사양 요약

| Stage | 파일 | 핵심 |
|---|---|---|
| Encoder | [models/tokenizer/encoder.py](models/tokenizer/encoder.py) | shared 1D-CNN, channels [32,64,128,256], stride 2×4 → AdaptiveAvgPool → Linear(latent=256). 옵션 L2-normalize + learnable scale. |
| Codebook | [models/tokenizer/codebook.py](models/tokenizer/codebook.py) | EMA VQ. `_ema_update` 안에서 cluster_sum/dw 를 `dist.all_reduce(SUM)` 으로 모아 모든 rank가 동일 EMA. Laplace smoothing. ST-grad. |
| Decoder | [models/tokenizer/decoder.py](models/tokenizer/decoder.py) | Linear → reshape (C, beat_length/16) → ConvTranspose1d ×4 (mirror). |
| Loss | [training/tokenizer/losses.py](training/tokenizer/losses.py) | `L_rec + α·L_vq + β·L_fid` ; L_fid = `‖∇x − ∇x̂‖²` (시간축 finite diff). |
| Embeddings | [models/context/embeddings.py](models/context/embeddings.py) | MorphologyEmbedding(`K+1`, +1 = MASK), LeadEmb(12), BeatPosEmb(20), RhythmMLP(3→128→128→d), GlobalContextCNN(12-ch 2D-CNN → AdaptiveAvgPool → d). |
| Transformer | [models/transformer/ecg_model.py](models/transformer/ecg_model.py) | Pre-LN `nn.TransformerEncoder`, sequence = `[g, T_{1,1},…,T_{N,12}]`, `out[:,0,:]` = CLS/global. |
| Masking | [training/pretrain/masking.py](training/pretrain/masking.py) | `lead_dropout(prob, min_leads)` → `mask_beat_tokens(ratio)` → `mask_rhythm_features(ratio)`. |
| Heads | [models/heads/mlm_head.py](models/heads/mlm_head.py) | MaskedBeatModelingHead(d→codebook), MaskedRhythmHead(d→3), ClassifierHead(`pooling: cls\|mean`). |

---

## 주요 하이퍼파라미터 (현재 main config)

| 항목 | 값 | 위치 |
|---|---|---|
| target_fs | 500Hz | tokenizer/pretrain configs |
| beat_length | 256 | (300 raw samples → 256 resample) |
| before/after_ms | 200 / 400 | |
| codebook K | **512** (메인), {256, 1024, 2048} ablation | |
| latent_dim | 256 | |
| EMA decay / commitment | 0.99 / 0.25 | |
| L_fid 가중치 β | 0.5 (gradient loss) | |
| Phase 1 batch / lr | 512/GPU · 3e-4 | |
| Phase 1 max_epochs | 100 (full) / 30 (200K subset) | |
| virtual_len train/val | 50M / 1.2M | streaming epoch size |
| max_beats_per_record | 10 (랜덤 샘플) | |
| Transformer d_model / heads / layers | 256 / 8 / 8 | |
| max_beats_per_lead (Phase 3) | 15 | |
| beat / rhythm mask ratio | 0.15 / 0.15 | |
| lead_dropout_prob / min_leads | 0.2 / 1 | |
| Phase 3 batch / lr / epochs | 128/GPU · 5e-4 · 100 | (DDP 7-GPU effective batch 896) |

---

## DDP 운용 노트

- **DDP launch**: `torchrun --standalone --nproc_per_node=$NPROC -m training.<phase>.train --config ...`. `master_port` 는 pretrain launcher가 randomize 해서 동시 실행 충돌 방지.
- **find_unused_parameters=True**: VQ codebook의 `embedding.weight` 가 EMA로만 갱신(autograd 그래프 미참여)이라 필요. Pretrain의 head들도 lead dropout으로 일부 step에서 unused 발생 가능.
- **EMA codebook 동기화**: `_ema_update` 내부에서 `dist.all_reduce` 로 cluster_sum / dw 를 합산. 따라서 `broadcast_buffers=False` (필요 없음).
- **OMP_NUM_THREADS**: 7 GPU × 8~24 worker → BLAS oversubscribe 방지를 위해 launcher에서 `OMP_NUM_THREADS=1~4` 강제.
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** — 대형 batch에서 fragmentation 완화.
- **자동 resume**: 모든 train.py가 `ckpt_dir/last.pt` 를 자동 로드. epoch/optimizer/scheduler/global_step/best_val_loss 모두 복구.

---

## 추론 / 사용 예시

### Tokenizer로 beat → indices

```python
import torch
from models.tokenizer.vqvae import VQVAE

ckpt = torch.load("checkpoints/tokenizer_heedb_full_cb512/best.pt", map_location="cpu")
tok  = VQVAE(ckpt["model_cfg"])
tok.load_state_dict(ckpt["model"]); tok.eval()

beats = torch.randn(B, 1, 256)            # z-score normalized beats
z_q, indices = tok.encode(beats)          # (B, 256), (B,)
recon = tok.decode_indices(indices)       # (B, 1, 256)
```

### Codebook 진단

```python
from utils.metrics import codebook_usage_rate, codebook_perplexity
print(codebook_usage_rate(indices, 512), codebook_perplexity(indices, 512))
```

### Lead-agnostic / reduced-lead inference

Pre-training에 `lead_dropout_prob=0.2` 가 들어가 있어 1-lead 추론도 안정적.

```python
indices  = indices[:, :, 1:2]            # Lead II 만
rr_feats = rr_feats[:, :, 1:2, :]
lead_ids = torch.tensor([1])             # II = index 1
out = model(indices, rr_feats, stft, lead_ids=lead_ids)
```

---

## 현재 학습 상태 (snapshot)

| 모델 | Config | 상태 |
|---|---|---|
| Tokenizer K=512 (full HEEDB, 100ep) | `vqvae_heedb_full_cb512.yaml` | ✅ 완료 — `checkpoints/tokenizer_heedb_full_cb512/best.pt` |
| Tokenizer K=512 (200K subset, 30ep) | `vqvae_heedb.yaml` | ✅ 완료 |
| Tokenizer K=256 ablation | `vqvae_heedb_full_cb256.yaml` | 🔄 진행 중 (`logs/ablation_runs/cb256_*.log`) |
| Tokenizer K=1024 / K=2048 ablation | `vqvae_heedb_full_cb{1024,2048}.yaml` | ⏳ 대기 (ablation 스크립트 큐) |
| Pretrain (Masked Beat Modeling, 100ep) | `masked_beat_heedb.yaml` | ✅ 완료 — `checkpoints/pretrain_heedb/best.pt` |
| Fine-tune (arrhythmia 등) | `arrhythmia.yaml` | ⏳ 데이터 준비 단계 |
