#!/bin/bash
# scripts/run_pretrain_ablation_v3.sh
#
# v3 Pre-training codebook ablation: cb{512,256,1024,2048} 순차 실행.
#
# v3 변경사항 (vs v2):
#   - record_mad 정규화 (V1↔V6 amplitude 보존)
#   - num_layers 8 → 12 (depth ↑, width 동일)
#   - mask_ratio warmup 0.15 → 0.5 over 50 epochs (linear)
#   - span_length 10 → 3
#   - lead_dropout 0.4 → 0.15, min_leads 6 → 10
#   - early stop: val_loss → val_acc_nontop (representation quality 직접 추적)
#   - SimCLR-style contrastive auxiliary loss (per-record CLS pair)
#
# 각 codebook별로 v3 tokenizer(checkpoints/tokenizer_heedb_full_cb{N}_v3/best.pt)를
# frozen으로 사용해서 pretrain을 처음부터 학습.
#
# 각 run의 ckpt/log는 분리 저장됨:
#   checkpoints/pretrain_heedb_cb{256,512,1024,2048}_v3/{best,last,epoch_*}.pt
#   logs/pretrain_heedb_cb{256,512,1024,2048}_v3/
#
# Usage:
#   nohup ./scripts/run_pretrain_ablation_v3.sh > pretrain_ablation_v3.log 2>&1 &
#
# Override:
#   GPUS=0,1,2,3,4      ./scripts/run_pretrain_ablation_v3.sh       # GPU 지정 (default: 0~4, 5개)
#   ONLY=cb512          ./scripts/run_pretrain_ablation_v3.sh       # 단일 config만 실행 (쉼표 구분)
#   SKIP=cb256          ./scripts/run_pretrain_ablation_v3.sh       # 특정 config 제외 (쉼표 구분)
#   FORCE_RESTART=1     ./scripts/run_pretrain_ablation_v3.sh       # 완료 stamp 무시 재실행
#                                                                 (train.py는 ckpt_dir/last.pt 에서
#                                                                  자동 resume → 학습 이어짐)

set -euo pipefail

# ── 사용할 GPU (0~4, 5개) ─────────────────────────────────────────────────────
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

# DataLoader worker × GPU 수 만큼 BLAS thread가 터지는 것 억제
# (5 GPU × 8 worker = 40 워커. 128 CPU 기준 충분 여유)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# ── 학습 대상 (사용자 지정 순서: 512 → 256 → 1024 → 2048) ───────────────────
ALL_CONFIGS=(
  "cb512:configs/pretrain/masked_beat_heedb_cb512_v3.yaml"
  "cb256:configs/pretrain/masked_beat_heedb_cb256_v3.yaml"
  "cb1024:configs/pretrain/masked_beat_heedb_cb1024_v3.yaml"
  "cb2048:configs/pretrain/masked_beat_heedb_cb2048_v3.yaml"
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
  echo "[pretrain-ablation] no configs to run after ONLY/SKIP filter"; exit 0
fi

# ── 로그/상태 디렉토리 ────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/pretrain_ablation_runs_v3"
STAMP_DIR="$LOG_ROOT/_stamps"
mkdir -p "$LOG_ROOT" "$STAMP_DIR"

FORCE_RESTART=${FORCE_RESTART:-0}

echo "============================================================"
echo "  ECG-FM  |  Pretrain Codebook Ablation v3"
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

  # ── tokenizer ckpt 사전 확인 ───────────────────────────────────────────────
  TOK_CKPT=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CFG'))['tokenizer']['ckpt'])")
  if [ ! -f "$TOK_CKPT" ]; then
    echo "[FAIL] $TAG: tokenizer checkpoint not found: $TOK_CKPT"
    echo "       Phase 1 먼저 완료되어야 함."
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
    echo "       train.py는 last.pt에서 자동 resume → 같은 명령으로 재실행하면 이어짐."
    exit "$rc"
  fi

  # best.pt 존재 확인 후 stamp
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
  printf "    %-7s → %s/best.pt\n" "$TAG" "$CKPT_DIR"
done
echo "============================================================"
