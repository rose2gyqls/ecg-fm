#!/bin/bash
# scripts/run_pretrain.sh
# Phase 3: ECG Foundation Model Pre-training (Masked Beat Modeling)
#
# Usage:
#   bash scripts/run_pretrain.sh
#   bash scripts/run_pretrain.sh configs/pretrain/masked_beat_base.yaml

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbkim

CONFIG=${1:-configs/pretrain/masked_beat_base.yaml}

echo "========================================"
echo "  ECG-FM  |  Phase 3: Pre-training"
echo "  Config : $CONFIG"
echo "  Time   : $(date)"
echo "========================================"

# tokenizer checkpoint가 존재하는지 확인
TOK_CKPT=$(python -c "
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

python -m training.pretrain.train --config "$CONFIG"

# 멀티 GPU
# torchrun --nproc_per_node=4 \
#   -m training.pretrain.train --config "$CONFIG"
