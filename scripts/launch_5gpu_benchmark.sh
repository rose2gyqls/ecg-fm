#!/usr/bin/env bash
# Launch 5 concurrent benchmark finetune jobs (one codebook size per GPU 0-4).
#
# Each job runs:  1 SIZE × 16 H5 TASKS × 3 EVAL_MODES = 48 (task, mode) tuples
# sequentially on its own GPU. All 5 GPUs train independently in parallel.
#
# Verified throughput (PTB-XL, B=128, 5 GPUs concurrent):
#   ~350 samples/s per GPU at 90-100% util — caching keeps GPU saturated.
#
# Usage:
#   bash scripts/launch_5gpu_benchmark.sh                         # all 16 tasks
#   TASKS="ptbxl_super chapman" bash scripts/launch_5gpu_benchmark.sh
#   EVAL_MODES="finetune_linear" bash scripts/launch_5gpu_benchmark.sh
#   EXCLUDE_CODE15=1 bash scripts/launch_5gpu_benchmark.sh        # skip CODE-15
#
# Stop:
#   pkill -f run_benchmark_finetune.sh    # all 5 jobs
#   kill <pid>                            # specific job (PIDs printed below)

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/benchmark

TS=$(date +%Y%m%d_%H%M%S)

# ── Required env (defaults work on this server) ──────────────────────────
export ECG_FM_HB_CACHE="${ECG_FM_HB_CACHE:-/home1/irteam/local-node-d/hbkimi/.cache/ecg_fm_hb_v4}"
export ECG_DATA_ROOT="${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}"
export ECG_FM_HB_REPO="${ECG_FM_HB_REPO:-$(pwd)}"

# ── Knobs ────────────────────────────────────────────────────────────────
EVAL_MODES_DEFAULT="linear_probe attention_probe finetune_linear"
export EVAL_MODES="${EVAL_MODES:-${EVAL_MODES_DEFAULT}}"

# Default 16 H5 tasks. echonext omitted (NumPy loader → cache miss, falls back
# to live preprocessing; run separately if needed).
TASKS_DEFAULT="ptbxl_super ptbxl_sub ptbxl_diag ptbxl_form ptbxl_rhythm ptbxl_all chapman chapman_rhythm cpsc2018 cpsc_extra ningbo georgia ptb code15 sph_diag zzu_pecg"
if [[ "${EXCLUDE_CODE15:-0}" -eq 1 ]]; then
    TASKS_DEFAULT="${TASKS_DEFAULT// code15/}"
fi
export TASKS="${TASKS:-${TASKS_DEFAULT}}"

# Pretrain + benchmark repos (forwarded by run_benchmark_finetune.sh).
export EPOCHS="${EPOCHS:-100}"
export LR="${LR:-1e-3}"
export CKPT_BASENAME="${CKPT_BASENAME:-best.pt}"
export BENCHMARK_REPO="${BENCHMARK_REPO:-/home/irteam/local-node-d/hbkimi/benchmark}"

# ── GPU → codebook size mapping ──────────────────────────────────────────
declare -A GPU_SIZE=(
    [0]=128
    [1]=256
    [2]=512
    [3]=1024
    [4]=2048
)

# Validate cache exists
if [[ ! -d "${ECG_FM_HB_CACHE}" ]] || [[ -z "$(ls -A "${ECG_FM_HB_CACHE}" 2>/dev/null)" ]]; then
    echo "[error] cache empty or missing: ${ECG_FM_HB_CACHE}" >&2
    echo "[error] run scripts/precompute_benchmark_cache.py first" >&2
    exit 2
fi

n_cache=$(ls "${ECG_FM_HB_CACHE}" | wc -l)
echo "[info] cache: ${ECG_FM_HB_CACHE} (${n_cache} files)"
echo "[info] tasks: ${TASKS}"
echo "[info] modes: ${EVAL_MODES}"
echo "[info] timestamp: ${TS}"
echo

PIDS=()
for gpu in 0 1 2 3 4; do
    size=${GPU_SIZE[$gpu]}
    ckpt="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${size}_v4/${CKPT_BASENAME}"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] GPU ${gpu} cb${size}: ${ckpt} not found"
        continue
    fi

    log="logs/benchmark/gpu${gpu}_cb${size}_${TS}.log"

    CUDA_VISIBLE_DEVICES=${gpu} SIZES=${size} \
        nohup bash scripts/run_benchmark_finetune.sh > "${log}" 2>&1 &
    pid=$!
    disown $pid
    PIDS+=($pid)
    echo "[launch] GPU ${gpu}  cb${size}  pid=${pid}  log=${log}"
done

echo
echo "[done] ${#PIDS[@]} jobs running. PIDs: ${PIDS[*]}"
cat <<EOF

# Monitor all 5 logs:
tail -f logs/benchmark/gpu*_${TS}.log

# Live GPU utilization:
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'

# Aggregate progress (completed runs):
ls -d ${BENCHMARK_REPO}/results/*ecg_fm_hb*/*/ 2>/dev/null | wc -l

# Per-tag latest epoch:
ls -t ${BENCHMARK_REPO}/results/*ecg_fm_hb*/*.log 2>/dev/null | head | xargs -I{} sh -c 'echo "=== {} ==="; tail -3 {}'

# Stop everything:
pkill -f run_benchmark_finetune.sh
EOF
