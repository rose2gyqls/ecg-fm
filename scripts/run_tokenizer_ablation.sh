#!/bin/bash
# scripts/run_tokenizer_ablation.sh
#
# VQ-VAE Beat Tokenizer ablation: codebook size 256 → 1024 → 2048 순차 실행.
# (cb512는 이미 학습 완료 → checkpoints/tokenizer_heedb_full_cb512/best.pt)
#
# 각 codebook 크기별 best 모델은 자동으로 분리 저장됨:
#   checkpoints/tokenizer_heedb_full_cb{256,1024,2048}/best.pt
#
# Usage:
#   nohup ./scripts/run_tokenizer_ablation.sh > ablation.log 2>&1 &
#
# Override:
#   GPUS=0,1,2,3,4   ./scripts/run_tokenizer_ablation.sh         # GPU 지정
#   ONLY=cb1024      ./scripts/run_tokenizer_ablation.sh         # 단일 config만 실행
#   SKIP=cb256       ./scripts/run_tokenizer_ablation.sh         # 특정 config 제외 (쉼표 구분)
#   FORCE_RESTART=1  ./scripts/run_tokenizer_ablation.sh         # 완료 stamp 무시하고 재실행

set -euo pipefail

# ── 사용할 GPU (0~4) ──────────────────────────────────────────────────────────
GPUS=${GPUS:-0,1,2,3,4}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

# ── 경로 / 환경 ───────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HBKIM_BIN=${HBKIM_BIN:-/home/irteam/local-node-d/_conda/envs/hbkim/bin}
export PATH="$HBKIM_BIN:$PATH"
PY="$HBKIM_BIN/python"
TORCHRUN="$HBKIM_BIN/torchrun"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# DataLoader worker가 BLAS thread를 oversubscribe하지 않도록 제한
# (5 GPU × 24 worker = 120 워커, 128 CPU 중 8코어 시스템 여유)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# NCCL 안정성/성능
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
# (필요 시 다음 항목 활성화)
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=0

# ── 학습 대상 (cb512는 이미 완료 → 제외) ──────────────────────────────────────
ALL_CONFIGS=(
  "cb256:configs/tokenizer/vqvae_heedb_full_cb256.yaml"
  "cb1024:configs/tokenizer/vqvae_heedb_full_cb1024.yaml"
  "cb2048:configs/tokenizer/vqvae_heedb_full_cb2048.yaml"
)

# ── 필터링 (ONLY / SKIP) ──────────────────────────────────────────────────────
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
  echo "[ablation] no configs to run after ONLY/SKIP filter"; exit 0
fi

# ── 로그/상태 디렉토리 ────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/ablation_runs"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  Tokenizer Codebook Ablation"
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
    echo "       set FORCE_RESTART=1 to rerun from scratch (or delete the stamp file)"
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
    echo "       train.py는 last.pt에서 자동 resume 가능 → 같은 명령으로 재실행하면 이어짐."
    exit "$rc"
  fi

  # best.pt 존재 확인 후 stamp
  CKPT_DIR=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  BEST="$CKPT_DIR/best.pt"
  if [ ! -f "$BEST" ]; then
    echo "[WARN] $TAG: best.pt not found at $BEST (학습은 끝났지만 best 저장 실패?)"
  fi

  date > "$STAMP"
  echo "[done] $TAG  elapsed=${dt}s  best=$BEST"
done

t_total=$(( $(date +%s) - t_all ))
echo
echo "============================================================"
echo "  All ablation runs complete.  total elapsed=${t_total}s"
echo "  Best checkpoints:"
for entry in "${ALL_CONFIGS[@]}"; do
  TAG="${entry%%:*}"; CFG="${entry#*:}"
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  printf "    %-7s → %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "    cb512   → checkpoints/tokenizer_heedb_full_cb512/best.pt   (이미 완료)"
echo "============================================================"
