#!/bin/bash
# scripts/run_finetune.sh
# Phase 4: Downstream Fine-tuning
#
# Usage:
#   bash scripts/run_finetune.sh arrhythmia
#   bash scripts/run_finetune.sh mi
#   bash scripts/run_finetune.sh heart_failure
#   bash scripts/run_finetune.sh configs/finetune/custom.yaml   (직접 경로 지정)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbkim

TASK=${1:-arrhythmia}

# 태스크명이 yaml 경로가 아니면 자동으로 경로 구성
if [[ "$TASK" == *.yaml ]]; then
  CONFIG="$TASK"
else
  CONFIG="configs/finetune/${TASK}.yaml"
fi

echo "========================================"
echo "  ECG-FM  |  Phase 4: Fine-tuning"
echo "  Task   : $TASK"
echo "  Config : $CONFIG"
echo "  Time   : $(date)"
echo "========================================"

# pretrained checkpoint 확인
PT_CKPT=$(python -c "
import yaml
with open('$CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg['base_pretrain_ckpt'])
")
if [ ! -f "$PT_CKPT" ]; then
  echo "[ERROR] Pretrained checkpoint not found: $PT_CKPT"
  echo "        Run Phase 3 first:  bash scripts/run_pretrain.sh"
  exit 1
fi
echo "  Pretrain ckpt: $PT_CKPT  ✓"

python -m training.finetune.train --config "$CONFIG"
