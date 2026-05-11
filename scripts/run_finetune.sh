#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HBKIM_BIN=${HBKIM_BIN:-}
if [ -n "$HBKIM_BIN" ]; then
  export PATH="$HBKIM_BIN:$PATH"
  PY="$HBKIM_BIN/python"
else
  PY=${PYTHON:-python}
fi

TASK=${1:-arrhythmia}
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
PT_CKPT=$("$PY" -c "
import yaml
with open('$CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg['base_pretrain_ckpt'])
")
if [ ! -f "$PT_CKPT" ]; then
  echo "[ERROR] Pretrained checkpoint not found: $PT_CKPT"
  echo "        Run Phase 3 first:  bash scripts/run_pretrain.sh"
  exit 1
fi
echo "  Pretrain ckpt: $PT_CKPT  ok"

"$PY" -m training.finetune.train --config "$CONFIG"
