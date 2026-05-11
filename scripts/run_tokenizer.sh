#!/bin/bash
# scripts/run_tokenizer.sh
# Phase 1: VQ-VAE Beat Tokenizer 학습
#
# Default: v4 cb1024 tokenizer config.
#
# Usage:
#   ./scripts/run_tokenizer.sh
#   ./scripts/run_tokenizer.sh configs/tokenizer/vqvae_heedb_full_cb512_v4.yaml
#
# GPU 번호는 아래 GPUS 변수를 직접 고쳐서 쓴다. NPROC은 GPUS 개수에서 자동 계산.
# env로 override도 가능: GPUS=0,1 ./scripts/run_tokenizer.sh configs/...yaml

set -e

# ── 사용할 GPU 번호 ────────────────────────────────────────────────────────────
# 쉼표로 구분. 학습 시 여기만 고치면 됨. (env GPUS=... 로 override 가능)
GPUS=${GPUS:-0,1,2,3,4,5,6}

# GPUS 개수로부터 DDP world size(nproc) 자동 계산
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional: point HBKIM_BIN at a specific environment's bin directory.
HBKIM_BIN=${HBKIM_BIN:-}
if [ -n "$HBKIM_BIN" ]; then
  export PATH="$HBKIM_BIN:$PATH"
  PY="$HBKIM_BIN/python"
  TORCHRUN="$HBKIM_BIN/torchrun"
else
  PY=${PYTHON:-python}
  TORCHRUN=${TORCHRUN:-torchrun}
fi

CONFIG=${1:-configs/tokenizer/vqvae_heedb_full_cb1024_v4.yaml}
RESUME=${RESUME:-}    # 명시 경로. 비우면 ckpt_dir/last.pt 자동 로드.

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

EXTRA_ARGS=()
if [ -n "$RESUME" ]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi

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
    -m training.tokenizer.train --config "$CONFIG" "${EXTRA_ARGS[@]}"
else
  "$PY" -m training.tokenizer.train --config "$CONFIG" "${EXTRA_ARGS[@]}"
fi
