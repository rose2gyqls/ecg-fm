#!/bin/bash
# scripts/run_tokenizer_ablation_v4.sh
#
# v4 Tokenizer ablation (cb128/256/512/1024/2048).
# v4 변경사항 (vs v3):
#   - data.preprocessing.resampler.compute_record_norm_stats:
#       per-lead p75 → median across leads. min_scale 0.01 → 0.05.
#       outlier record amplification 차단 (v3에서 batch loss=7+ 튀던 원인).
#   - data.record_mad_clip=8.0 (post-norm hard cap).
#   - codebook usage entropy regularizer (loss.delta=0.05) 추가.
#       V1↔V6가 같은 코드로 collapse하던 문제를 entropy bonus로 깨준다.
#   - codebook ema_decay 0.99 → 0.97 (half-life 69 → 23 step). v3에서
#     encoder의 점진적 drift를 EMA가 못 따라잡아 loss_vq 우상향했던
#     desync 패턴을 직접 차단. commitment_cost 0.1 → 0.25 로 encoder를
#     code 쪽으로 더 강하게 잡아당겨 desync 자체를 줄임.
#   - restart_dead_every 500 → 2000, restart_dead_threshold 1.0 → 0.1
#     (codebook restart shock 완화).
#   - loss.gamma 0.3 → 0.15 (spec loss 비중 절반).
#   - training.warmup_epochs=5 가 실제 LinearLR로 적용됨.
#   - training.best_window=5 — sliding-window val_loss로 best.pt 선택.
#
# 결과는 checkpoints/tokenizer_heedb_full_cb*_v4/ 아래에 분리 저장.
# v1/v2/v3 체크포인트는 그대로 유지.
#
# Usage:
#   nohup ./scripts/run_tokenizer_ablation_v4.sh > tokenizer_v4.log 2>&1 &
#
# Override:
#   GPUS=0,1,2,3,4   ./scripts/run_tokenizer_ablation_v4.sh
#   ONLY=cb512       ./scripts/run_tokenizer_ablation_v4.sh
#   SKIP=cb2048      ./scripts/run_tokenizer_ablation_v4.sh
#   FORCE_RESTART=1  ./scripts/run_tokenizer_ablation_v4.sh

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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

ALL_CONFIGS=(
  "cb128:configs/tokenizer/vqvae_heedb_full_cb128_v4.yaml"
  "cb256:configs/tokenizer/vqvae_heedb_full_cb256_v4.yaml"
  "cb512:configs/tokenizer/vqvae_heedb_full_cb512_v4.yaml"
  "cb1024:configs/tokenizer/vqvae_heedb_full_cb1024_v4.yaml"
  "cb2048:configs/tokenizer/vqvae_heedb_full_cb2048_v4.yaml"
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
  echo "[ablation_v4] no configs to run after ONLY/SKIP filter"; exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/ablation_runs_v4"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  Tokenizer Codebook Ablation v4"
echo "  GPUs   : $GPUS  (nproc=$NPROC)"
echo "  Configs: $(printf '%s ' "${CONFIGS[@]%%:*}")"
echo "  Start  : $(date)"
echo "============================================================"

t_all=$(date +%s)

for entry in "${CONFIGS[@]}"; do
  TAG="${entry%%:*}"
  CFG="${entry#*:}"
  STAMP="$STAMP_DIR/${TAG}.done"
  LOGFILE="$LOG_ROOT/${TAG}_${TS}.log"

  if [ -f "$STAMP" ] && [ "$FORCE_RESTART" != "1" ]; then
    echo "[skip] $TAG  ← already complete (stamp: $STAMP)"
    continue
  fi

  echo
  echo "------------------------------------------------------------"
  echo "  >>> $TAG   ($CFG)"
  echo "  log : $LOGFILE"
  echo "  time: $(date)"
  echo "------------------------------------------------------------"

  t0=$(date +%s)
  set +e
  "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
    -m training.tokenizer.train --config "$CFG" 2>&1 | tee "$LOGFILE"
  rc=${PIPESTATUS[0]}
  set -e
  dt=$(( $(date +%s) - t0 ))

  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] $TAG  rc=$rc  elapsed=${dt}s  log=$LOGFILE"
    echo "       train.py는 last.pt에서 자동 resume 가능."
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
echo "  All v4 ablation runs complete.  total elapsed=${t_total}s"
echo "  Best checkpoints:"
for entry in "${ALL_CONFIGS[@]}"; do
  TAG="${entry%%:*}"; CFG="${entry#*:}"
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  printf "    %-7s → %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "============================================================"
