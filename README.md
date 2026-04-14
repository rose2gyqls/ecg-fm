# ECG Foundation Model — Hierarchical Beat-to-Sequence Framework

Beat-level VQ-VAE tokenization + Masked Beat Modeling 기반 ECG Foundation Model 구현체.

---

## 아키텍처 요약

```
Stage 1  Beat Tokenization   : 12-lead beat → Shared 1D-CNN → VQ Codebook → index z
Stage 2  Context Injection   : T_{i,j} = Emb(z) + RhythmMLP(p) + LeadEmb(j) + PosEmb(i)
Stage 3  Sequence Modeling   : [g, T_{1,1}, ..., T_{N,12}] → Transformer Encoder → [CLS]
```

---

## 프로젝트 구조

```
ecg-fm/
├── configs/
│   ├── tokenizer/vqvae_base.yaml
│   ├── pretrain/masked_beat_base.yaml
│   └── finetune/{arrhythmia,mi,heart_failure}.yaml
│
├── data/
│   ├── preprocessing/
│   │   ├── beat_segmentor.py     # R-peak 검출 + beat 추출 + RR features
│   │   ├── resampler.py          # 256샘플 리샘플 + z-score 정규화
│   │   └── stft_extractor.py     # 10초 ECG → log-STFT (12, F, T')
│   └── datasets/
│       ├── beat_dataset.py       # Phase 1: (1, 256) 단일 beat
│       ├── ecg_dataset.py        # Phase 3: beats + rr_feats + stft
│       └── finetune_dataset.py   # Phase 4: ecg_dataset + label
│
├── models/
│   ├── tokenizer/
│   │   ├── encoder.py            # Shared 1D-CNN  (B,1,256) → (B,256)
│   │   ├── codebook.py           # EMA VQ Codebook K=512
│   │   ├── decoder.py            # ConvTranspose1D (B,256) → (B,1,256)
│   │   └── vqvae.py              # 통합 VQ-VAE
│   ├── context/
│   │   └── embeddings.py         # LeadEmb / PosEmb / RhythmMLP / GlobalContextCNN
│   ├── transformer/
│   │   └── ecg_model.py          # ECGFoundationModel (Stage2+3)
│   └── heads/
│       └── mlm_head.py           # MaskedBeatModelingHead / MaskedRhythmHead / ClassifierHead
│
├── training/
│   ├── tokenizer/
│   │   ├── train.py              # Phase 1 학습 루프
│   │   └── losses.py             # L_rec + α·L_vq + β·L_fiducial
│   ├── pretrain/
│   │   ├── train.py              # Phase 3 Masked Beat Modeling
│   │   └── masking.py            # beat / rhythm / lead dropout 마스킹
│   └── finetune/
│       └── train.py              # Phase 4 + AUROC/F1 평가
│
├── utils/
│   ├── checkpointing.py          # save / load checkpoint
│   ├── logging_utils.py          # CSV + 콘솔 MetricLogger
│   └── metrics.py                # codebook 진단 + clf metrics
│
├── scripts/
│   ├── run_tokenizer.sh
│   ├── run_pretrain.sh
│   └── run_finetune.sh
│
├── environment.yaml
└── README.md
```

---

## 환경 설정

```bash
# 기존 hbkim conda 환경에 패키지 추가
conda activate hbkim
pip install neurokit2 wfdb h5py scikit-learn
```

---

## 데이터 준비

### 권장 파일 포맷 (npy dict)

```python
import numpy as np

# 각 레코드를 dict로 저장
np.save("data_dir/train/record_001.npy", {
    "signal": ecg_array,     # (12, T) float32, T = fs * 10
    "rpeaks": rpeak_indices, # (N,) int, 선택
    "label":  0,             # int, fine-tune 시 필요
})
```

### 디렉토리 구조

```
data_dir/
├── train/
│   ├── record_001.npy
│   └── ...
└── val/
    └── ...
```

### config에서 경로 수정

```yaml
# configs/tokenizer/vqvae_base.yaml
data:
  data_dir: /home/irteam/local-node-d/hbkimi/ecg-fm/data_dir
```

---

## 학습 실행

### Phase 1: Beat Tokenizer

```bash
cd /home/irteam/local-node-d/hbkimi/ecg-fm
conda activate hbkim

bash scripts/run_tokenizer.sh
# 또는
python -m training.tokenizer.train --config configs/tokenizer/vqvae_base.yaml
```

완료 후 `checkpoints/tokenizer/best.pt` 생성 확인.

**모니터링 지표:**
- `loss_rec`: 낮을수록 복원 품질 좋음
- `perplexity`: codebook 크기(512)에 가까울수록 고른 분포
- `loss_fid`: gradient loss, QRS 등 형태 보존 확인

### Phase 3: Pre-training

```bash
# configs/pretrain/masked_beat_base.yaml의 tokenizer.ckpt 경로 확인 후
bash scripts/run_pretrain.sh
```

### Phase 4: Fine-tuning

```bash
# 부정맥
bash scripts/run_finetune.sh arrhythmia

# 심근경색
bash scripts/run_finetune.sh mi

# 커스텀 config
bash scripts/run_finetune.sh configs/finetune/custom.yaml
```

---

## 주요 하이퍼파라미터 가이드

| 항목 | 기본값 | 조정 팁 |
|------|--------|---------|
| codebook K | 512 | perplexity < 100이면 K 줄이기, collapse 시 K 늘리기 |
| beat_length | 256 | 500Hz × 200ms~400ms 범위 |
| beat_mask_ratio | 0.15 | 0.1~0.3 실험 |
| lead_dropout_prob | 0.2 | wearable 적용 목적이면 0.5까지 |
| d_model | 256 | 데이터 많으면 512로 확장 |
| num_layers | 8 | GPU 메모리에 따라 조정 |

---

## Codebook 진단

```python
from utils.metrics import codebook_usage_rate, codebook_perplexity

# 학습 중 batch indices로 확인
usage = codebook_usage_rate(indices, codebook_size=512)
ppl   = codebook_perplexity(indices, codebook_size=512)
print(f"Usage: {usage:.2%}  Perplexity: {ppl:.1f}")
```

**Codebook collapse 징후:** usage < 10%, perplexity < 50
→ EMA decay 낮추기 (0.99 → 0.95), learning rate 확인

---

## Lead-Agnostic Inference (1-lead / reduced-lead)

```python
# 1-lead (Lead II만 있는 경우)
indices   = indices[:, :, 1:2]    # (B, N, 1)
rr_feats  = rr_feats[:, :, 1:2, :]
lead_ids  = torch.tensor([1])      # Lead II = index 1

output = model(indices, rr_feats, stft, lead_ids=lead_ids)
```

pre-training 시 lead dropout 덕분에 1-lead에서도 안정적으로 동작.
