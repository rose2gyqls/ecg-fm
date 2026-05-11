#!/bin/bash
# scripts/run_pretrain.sh
# Phase 3: ECG Foundation Model Pre-training (Masked Beat Modeling)
#
# Default: v4 cb1024 pre-training config on GPU 0-6 (7개) DDP
# Usage:
#   bash scripts/run_pretrain.sh                                  # GPU 0,1,2,3,4,5,6
#   bash scripts/run_pretrain.sh configs/pretrain/masked_beat_heedb_cb512_v4.yaml
#   GPUS=0,1 bash scripts/run_pretrain.sh                         # custom GPUs
#   GPUS=0   bash scripts/run_pretrain.sh                         # single GPU

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HBKIM_BIN=${HBKIM_BIN:-}
if [ -n "$HBKIM_BIN" ]; then
  export PATH="$HBKIM_BIN:$PATH"
  PY="$HBKIM_BIN/python"
  TORCHRUN="$HBKIM_BIN/torchrun"
else
  PY=${PYTHON:-python}
  TORCHRUN=${TORCHRUN:-torchrun}
fi

CONFIG=${1:-configs/pretrain/masked_beat_heedb_cb1024_v4.yaml}
GPUS=${GPUS:-0,1,2,3,4,5,6}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "========================================"
echo "  ECG-FM  |  Phase 3: Pre-training"
echo "  Config : $CONFIG"
echo "  GPUs   : $GPUS  (nproc=$NPROC)"
echo "  Time   : $(date)"
echo "========================================"

# tokenizer ckpt 존재 확인
TOK_CKPT=$("$PY" -c "
import yaml
with open('$CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg['tokenizer']['ckpt'])
")
if [ ! -f "$TOK_CKPT" ]; then
  echo "[ERROR] Tokenizer checkpoint not found: $TOK_CKPT"
  echo "        Run Phase 1 first:  bash scripts/run_tokenizer.sh"
  exit 1
fi
echo "  Tokenizer: $TOK_CKPT  ✓"

export CUDA_VISIBLE_DEVICES=$GPUS
export NCCL_ASYNC_ERROR_HANDLING=1
# CPU oversubscription 방지 — DataLoader workers × GPU 수 만큼 thread가 생기는 것 억제
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [ "$NPROC" = "1" ]; then
    "$PY" -m training.pretrain.train --config "$CONFIG"
else
    # master_port를 randomize 해서 여러 학습이 동시 실행되어도 충돌 방지
    PORT=$((10000 + RANDOM % 50000))
    "$TORCHRUN" --nproc_per_node=$NPROC --master_port=$PORT \
        -m training.pretrain.train --config "$CONFIG"
fi
