#!/bin/bash

set -euo pipefail
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
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

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
ALL_CONFIGS=(
  "cb512:configs/pretrain/masked_beat_heedb_cb512_v4.yaml"
  "cb256:configs/pretrain/masked_beat_heedb_cb256_v4.yaml"
  "cb1024:configs/pretrain/masked_beat_heedb_cb1024_v4.yaml"
  "cb2048:configs/pretrain/masked_beat_heedb_cb2048_v4.yaml"
  "cb128:configs/pretrain/masked_beat_heedb_cb128_v4.yaml"
)
ONLY=${ONLY:-}
SKIP=${SKIP:-}
in_csv() { local needle="$1" csv="$2"; [[ ",$csv," == *",$needle,"* ]]; }

CONFIGS=()
for entry in "${ALL_CONFIGS[@]}"; do
  tag="${entry%%:*}"
  if [ -n "$ONLY" ] && ! in_csv "$tag" "$ONLY"; then continue; fi
  if [ -n "$SKIP" ] &&    in_csv "$tag" "$SKIP"; then continue; fi
  CONFIGS+=("$entry")
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "[pretrain-ablation-v4] no configs to run after ONLY/SKIP filter"; exit 0
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/pretrain_ablation_runs_v4"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  Pretrain Codebook Ablation v4"
echo "  GPUs   : $GPUS  (nproc=$NPROC)"
echo "  Configs: $(printf '%s ' "${CONFIGS[@]%%:*}")"
echo "  Start  : $(date)"
echo "  Python : $($PY -c 'import sys; print(sys.executable)')"
echo "  Torch  : $($PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "============================================================"

t_all=$(date +%s)

for entry in "${CONFIGS[@]}"; do
  TAG="${entry%%:*}"
  CFG="${entry#*:}"
  STAMP="$STAMP_DIR/${TAG}.done"
  LOGFILE="$LOG_ROOT/${TAG}_${TS}.log"

  if [ -f "$STAMP" ] && [ "$FORCE_RESTART" != "1" ]; then
    echo "[skip] $TAG  already complete (stamp: $STAMP)"
    echo "       set FORCE_RESTART=1 to rerun from scratch (or delete the stamp file)"
    continue
  fi
  TOK_CKPT=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['tokenizer']['ckpt'])")
  if [ ! -f "$TOK_CKPT" ]; then
    echo "[FAIL] $TAG: tokenizer checkpoint not found: $TOK_CKPT"
    echo "       Run tokenizer training before pretraining."
    exit 1
  fi

  echo
  echo "------------------------------------------------------------"
  echo "  >>> $TAG   ($CFG)"
  echo "  tok : $TOK_CKPT"
  echo "  log : $LOGFILE"
  echo "  time: $(date)"
  echo "------------------------------------------------------------"

  PORT=$((10000 + RANDOM % 50000))

  t0=$(date +%s)
  set +e
  if [ "$NPROC" = "1" ]; then
    "$PY" -m training.pretrain.train --config "$CFG" 2>&1 | tee "$LOGFILE"
  else
    "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$PORT" \
      -m training.pretrain.train --config "$CFG" 2>&1 | tee "$LOGFILE"
  fi
  rc=${PIPESTATUS[0]}
  set -e
  dt=$(( $(date +%s) - t0 ))

  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] $TAG  rc=$rc  elapsed=${dt}s  log=$LOGFILE"
    echo "       train.py auto-resumes from last.pt; rerun the same command to continue."
    exit "$rc"
  fi
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  BEST="$CKPT_DIR/best.pt"
  if [ ! -f "$BEST" ]; then
    echo "[WARN] $TAG: best.pt not found at $BEST"
  fi

  date > "$STAMP"
  echo "[done] $TAG  elapsed=${dt}s  best=$BEST"
done

t_total=$(( $(date +%s) - t_all ))
echo
echo "============================================================"
echo "  All pretrain ablation runs complete.  total elapsed=${t_total}s"
echo "  Best checkpoints:"
for entry in "${ALL_CONFIGS[@]}"; do
  TAG="${entry%%:*}"; CFG="${entry#*:}"
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  printf "    %-7s -> %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "============================================================"
