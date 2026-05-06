#!/bin/bash
# scripts/run_pretrain_ablation_v5.sh
#
# v5 (MoRyECG) Pre-training codebook ablation: cb{128,256,512,1024,2048}.
#
# v5 변경사항 (vs v4):
#   - Architecture: flat-seq Transformer → MoRyECG (intra-beat cross-lead +
#     inter-beat rhythm w/ STFT-conditioned [GLOB], shared RR-aware bias)
#   - STFT leakage 차단:
#       (a) lead-dropout zeroes dropped-lead STFT channels (v4와 동일)
#       (b) NEW: beat-aligned STFT-time bin zero — masked beat의 raw sample
#           window를 덮는 모든 STFT frame을 0으로
#   - Slim STFT encoder ([8, 16, 32]) + STFT bin dropout
#   - Pairwise RR additive bias (init_zero, all blocks share the MLP,
#     per-head slopes)
#   - Tokenizer는 v4 그대로 (cb{N}_v4/best.pt)
#   - Loss: MLM + Contrastive (RR/Fid head 코드 유지하되 weight=0 default)
#
# 각 run의 ckpt/log는 분리 저장:
#   checkpoints/pretrain_heedb_cb{N}_v5/{best,last,epoch_*}.pt
#   logs/pretrain_heedb_cb{N}_v5/
#
# Usage:
#   nohup ./scripts/run_pretrain_ablation_v5.sh > pretrain_ablation_v5.log 2>&1 &
#
# Override:
#   GPUS=0,1,2,3        ./scripts/run_pretrain_ablation_v5.sh   # default 4-GPU
#   ONLY=cb1024         ./scripts/run_pretrain_ablation_v5.sh   # 단일 config
#   SKIP=cb128          ./scripts/run_pretrain_ablation_v5.sh   # 특정 config 제외
#   FORCE_RESTART=1     ./scripts/run_pretrain_ablation_v5.sh   # 완료 stamp 무시 재실행

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

# 학습 대상 (v4와 동일 순서: 1024 우선 검증 후 다른 size로 확장하기 좋게)
ALL_CONFIGS=(
  "cb1024:configs/pretrain/moryecg_heedb_cb1024_v5.yaml"
  "cb512:configs/pretrain/moryecg_heedb_cb512_v5.yaml"
  "cb256:configs/pretrain/moryecg_heedb_cb256_v5.yaml"
  "cb2048:configs/pretrain/moryecg_heedb_cb2048_v5.yaml"
  "cb128:configs/pretrain/moryecg_heedb_cb128_v5.yaml"
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
  echo "[pretrain-ablation-v5] no configs to run after ONLY/SKIP filter"; exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/pretrain_ablation_runs_v5"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  MoRyECG (v5) Pretrain Codebook Ablation"
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

  # tokenizer ckpt 사전 확인 (v5는 v4 tokenizer를 그대로 재사용)
  TOK_CKPT=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['tokenizer']['ckpt'])")
  if [ ! -f "$TOK_CKPT" ]; then
    echo "[FAIL] $TAG: v4 tokenizer checkpoint not found: $TOK_CKPT"
    echo "       v5 reuses v4 tokenizer (cb{N}_v4/best.pt). Phase 1 v4가 먼저 완료되어야 함."
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
    "$PY" -m training.pretrain.train_v5 --config "$CFG" 2>&1 | tee "$LOGFILE"
  else
    "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$PORT" \
      -m training.pretrain.train_v5 --config "$CFG" 2>&1 | tee "$LOGFILE"
  fi
  rc=${PIPESTATUS[0]}
  set -e
  dt=$(( $(date +%s) - t0 ))

  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] $TAG  rc=$rc  elapsed=${dt}s  log=$LOGFILE"
    echo "       train_v5.py는 last.pt에서 자동 resume → 같은 명령으로 재실행하면 이어짐."
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
echo "  All v5 pretrain ablation runs complete.  total elapsed=${t_total}s"
echo "  Best checkpoints:"
for entry in "${ALL_CONFIGS[@]}"; do
  TAG="${entry%%:*}"; CFG="${entry#*:}"
  CKPT_DIR=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['ckpt_dir'])")
  printf "    %-7s → %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "============================================================"
