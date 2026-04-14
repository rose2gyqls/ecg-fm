#!/bin/bash
# scripts/run_tokenizer.sh
# Phase 1: VQ-VAE Beat Tokenizer 학습
#
# Usage:
#   bash scripts/run_tokenizer.sh                                     # 단일 GPU, 기본 config
#   bash scripts/run_tokenizer.sh configs/tokenizer/vqvae_heedb.yaml  # 단일 GPU, 지정 config
#   NPROC=2 GPUS=0,1 bash scripts/run_tokenizer.sh configs/tokenizer/vqvae_heedb.yaml  # DDP

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# hbkim 환경 python/torchrun 직접 지정 (conda activate가 다른 env를 먼저 잡는 문제 회피)
HBKIM_BIN=${HBKIM_BIN:-/home/irteam/local-node-d/_conda/envs/hbkim/bin}
export PATH="$HBKIM_BIN:$PATH"
PY="$HBKIM_BIN/python"
TORCHRUN="$HBKIM_BIN/torchrun"

CONFIG=${1:-configs/tokenizer/vqvae_base.yaml}
NPROC=${NPROC:-1}
GPUS=${GPUS:-0}

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "========================================"
echo "  ECG-FM  |  Phase 1: Beat Tokenizer"
echo "  Config : $CONFIG"
echo "  GPUs   : $GPUS  (nproc=$NPROC)"
echo "  Time   : $(date)"
echo "========================================"

echo "  Python : $($PY -c 'import sys; print(sys.executable)')"
echo "  Torch  : $($PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

if [ "$NPROC" -gt 1 ]; then
  "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
    -m training.tokenizer.train --config "$CONFIG"
else
  "$PY" -m training.tokenizer.train --config "$CONFIG"
fi
