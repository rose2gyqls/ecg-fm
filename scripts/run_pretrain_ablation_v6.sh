#!/bin/bash
# scripts/run_pretrain_ablation_v6.sh
#
# v6 (MoRyECG + GlobalRefineBlock) Pre-training codebook ablation.
#
# v6 changes vs v5 (informed by v4↔v5 finetune gap diagnosis):
#   Architecture:
#     - + GlobalRefineBlock: one final full-attention block over [g, H_flat]
#       to restore arbitrary token-to-token paths the factorization severed.
#   Loss (re-enabled to match v4):
#     - rhythm_weight 0 → 1.0
#     - fiducial_weight 0 → 1.0
#   STFT pathway (capacity restored):
#     - stft_channels [8,16,32] → [16,32,64]
#     - stft_dropout 0.1 → 0.0  (beat-aligned mask is the real leakage gate)
#   RR bias init:
#     - init_zero true → false (random init lets bias contribute from step 1)
#   Training schedule:
#     - early_stop_patience 10 → 30  (v4/v5 stopped too early)
#     - mask_warmup_epochs 50 → 25
#   Contrastive:
#     - weight 0.3 → 0.5,  temperature 0.1 → 0.07
#   Data:
#     - train_files_1m.txt (1.0M records, ~5× v5's 190k)
#       Closer to Chinchilla optimal for 76M-param model.
#
# Tokenizer is unchanged (v4 cb{N}_v4/best.pt frozen).
#
# Usage:
#   nohup ./scripts/run_pretrain_ablation_v6.sh > pretrain_ablation_v6.log 2>&1 &
#
# Override:
#   GPUS=0,1,2,3,4,5,6  ./scripts/run_pretrain_ablation_v6.sh   # 7 GPU default
#   ONLY=cb1024         ./scripts/run_pretrain_ablation_v6.sh
#   SKIP=cb128          ./scripts/run_pretrain_ablation_v6.sh
#   FORCE_RESTART=1     ./scripts/run_pretrain_ablation_v6.sh

set -euo pipefail

GPUS=${GPUS:-0,1,2,3,4,5,6}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HBKIM_BIN=${HBKIM_BIN:-/home/irteam/local-node-d/_conda/envs/hbkim/bin}
export PATH="$HBKIM_BIN:$PATH"
PY="$HBKIM_BIN/python"
TORCHRUN="$HBKIM_BIN/torchrun"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# 학습 대상 (cb1024 우선 검증 가능 순서)
ALL_CONFIGS=(
  "cb1024:configs/pretrain/moryecg_heedb_cb1024_v6.yaml"
  "cb512:configs/pretrain/moryecg_heedb_cb512_v6.yaml"
  "cb256:configs/pretrain/moryecg_heedb_cb256_v6.yaml"
  "cb2048:configs/pretrain/moryecg_heedb_cb2048_v6.yaml"
  "cb128:configs/pretrain/moryecg_heedb_cb128_v6.yaml"
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
  echo "[pretrain-ablation-v6] no configs to run after ONLY/SKIP filter"; exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/pretrain_ablation_runs_v6"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  MoRyECG v6 (with GlobalRefineBlock) Pretrain Ablation"
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
    echo "[skip] $TAG  ← already complete (stamp: $STAMP)"
    echo "       set FORCE_RESTART=1 to rerun (or delete the stamp file)"
    continue
  fi

  TOK_CKPT=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['tokenizer']['ckpt'])")
  if [ ! -f "$TOK_CKPT" ]; then
    echo "[FAIL] $TAG: v4 tokenizer checkpoint not found: $TOK_CKPT"
    echo "       v6 reuses v4 tokenizer (cb{N}_v4/best.pt)."
    exit 1
  fi

  echo
  echo "------------------------------------------------------------"
  echo "  >>> $TAG   ($CFG)"
  echo "  tok : $TOK_CKPT  (v4 frozen)"
  echo "  log : $LOGFILE"
  echo "  time: $(date)"
  echo "------------------------------------------------------------"

  PORT=$((10000 + RANDOM % 50000))

  t0=$(date +%s)
  set +e
  if [ "$NPROC" = "1" ]; then
    "$PY" -m training.pretrain.train_v6 --config "$CFG" 2>&1 | tee "$LOGFILE"
  else
    "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$PORT" \
      -m training.pretrain.train_v6 --config "$CFG" 2>&1 | tee "$LOGFILE"
  fi
  rc=${PIPESTATUS[0]}
  set -e
  dt=$(( $(date +%s) - t0 ))

  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] $TAG  rc=$rc  elapsed=${dt}s  log=$LOGFILE"
    echo "       train_v6.py auto-resumes from last.pt; rerun the same command to continue."
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
echo "  All v6 pretrain ablation runs complete.  total elapsed=${t_total}s"
echo "  Best checkpoints:"
for entry in "${ALL_CONFIGS[@]}"; do
  TAG="${entry%%:*}"; CFG="${entry#*:}"
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  printf "    %-7s → %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "============================================================"
