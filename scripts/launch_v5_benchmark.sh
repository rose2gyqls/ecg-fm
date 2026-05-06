#!/usr/bin/env bash
# Launch downstream benchmark finetune for v5 (MoRyECG) HEEDB pretrains across
# all available GPUs using a shared job queue.
#
# Why a queue (not 1-GPU-per-size as in launch_5gpu_benchmark.sh):
#   v5 has 5 codebook sizes but the host has 7 H200s. Pinning size→GPU leaves
#   GPUs 5-6 idle for the full run. A flock-based job queue spreads the
#   5 × |TASKS| × |EVAL_MODES| tuples across all 7 GPUs so every device stays
#   ~100% utilized for the entire duration. Single-job-per-GPU is already
#   compute-bound (~350 samples/s per H200 with caching), so we run exactly
#   one job per GPU at a time.
#
# Usage:
#   bash scripts/launch_v5_benchmark.sh                       # all sizes × tasks × modes
#   TASKS="ptbxl_super chapman" bash scripts/launch_v5_benchmark.sh
#   EVAL_MODES=finetune_linear bash scripts/launch_v5_benchmark.sh
#   GPUS="0 1 2 3 4 5 6" bash scripts/launch_v5_benchmark.sh   # explicit GPU set
#   SIZES="512 1024" bash scripts/launch_v5_benchmark.sh       # subset of sizes
#
# Stop:
#   pkill -f run_benchmark_finetune_one.sh    # current jobs
#   pkill -f launch_v5_benchmark.sh           # the launcher itself
#
# Inspect:
#   tail -f logs/benchmark_v5/queue.log
#   tail -f logs/benchmark_v5/gpu*_<TS>.log

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/benchmark_v5

TS=$(date +%Y%m%d_%H%M%S)
QUEUE_DIR=$(mktemp -d -t ecgfm_v5_queue.XXXXXX)
trap 'rm -rf "${QUEUE_DIR}"' EXIT

# ── Required env (defaults work on this server) ──────────────────────────
# v5 uses the v4 tokenizer (per moryecg_heedb_cb*_v5.yaml) and the same
# preprocessing pipeline (PREPROC_VERSION="v4" in src/encoders/ecg_fm_hb.py),
# so the v4 cache is reused as-is — saves re-precomputing 191 GB.
export ECG_FM_HB_CACHE="${ECG_FM_HB_CACHE:-/home1/irteam/local-node-d/hbkimi/.cache/ecg_fm_hb_v4}"
export ECG_DATA_ROOT="${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}"
export ECG_FM_HB_REPO="${ECG_FM_HB_REPO:-$(pwd)}"

# ── Knobs ────────────────────────────────────────────────────────────────
SIZES="${SIZES:-128 256 512 1024 2048}"
TASKS_DEFAULT="ptbxl_super ptbxl_sub ptbxl_diag ptbxl_form ptbxl_rhythm ptbxl_all chapman chapman_rhythm cpsc2018 cpsc_extra ningbo georgia ptb code15 sph_diag zzu_pecg"
if [[ "${EXCLUDE_CODE15:-0}" -eq 1 ]]; then
    TASKS_DEFAULT="${TASKS_DEFAULT// code15/}"
fi
TASKS="${TASKS:-${TASKS_DEFAULT}}"
EVAL_MODES="${EVAL_MODES:-linear_probe attention_probe finetune_linear}"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
CKPT_BASENAME="${CKPT_BASENAME:-best.pt}"
BENCHMARK_REPO="${BENCHMARK_REPO:-/home/irteam/local-node-d/hbkimi/benchmark}"

# All H200s on this host. Override GPUS=... to use a subset.
GPUS="${GPUS:-0 1 2 3 4 5 6}"

# Single shared OUT_ROOT so the benchmark's CSV aggregator finds every tuple.
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_REPO}/results/${TS}_ecg_fm_hb_v5}"
mkdir -p "${OUT_ROOT}"

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -d "${BENCHMARK_REPO}" ]]; then
    echo "[error] BENCHMARK_REPO not found: ${BENCHMARK_REPO}" >&2; exit 2
fi
if [[ ! -d "${ECG_FM_HB_CACHE}" ]] || [[ -z "$(ls -A "${ECG_FM_HB_CACHE}" 2>/dev/null)" ]]; then
    echo "[error] cache empty or missing: ${ECG_FM_HB_CACHE}" >&2; exit 2
fi

n_cache=$(ls "${ECG_FM_HB_CACHE}" | wc -l)
n_gpus=$(echo "${GPUS}" | wc -w)
echo "[info] timestamp : ${TS}"
echo "[info] gpus      : ${GPUS}  (n=${n_gpus})"
echo "[info] sizes     : ${SIZES}"
echo "[info] tasks     : ${TASKS}"
echo "[info] modes     : ${EVAL_MODES}"
echo "[info] epochs    : ${EPOCHS}   lr=${LR}"
echo "[info] cache     : ${ECG_FM_HB_CACHE} (${n_cache} files)"
echo "[info] out_root  : ${OUT_ROOT}"
echo

# ── Build job queue: one line per (size, task, mode) tuple ───────────────
# Skip tuples whose results already exist (idempotent re-runs).
QUEUE_FILE="${QUEUE_DIR}/jobs.txt"
LOCK_FILE="${QUEUE_DIR}/queue.lock"
QUEUE_LOG="logs/benchmark_v5/queue_${TS}.log"
: > "${QUEUE_FILE}"
: > "${LOCK_FILE}"

n_total=0; n_skip=0
for K in ${SIZES}; do
    CKPT="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${K}_v5/${CKPT_BASENAME}"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[skip-size] cb${K}_v5: checkpoint missing at ${CKPT}"
        continue
    fi
    for TASK in ${TASKS}; do
        for MODE in ${EVAL_MODES}; do
            tag="ecg_fm_hb_cb${K}_v5__${TASK}__${MODE}"
            save_dir="${OUT_ROOT}/${tag}"
            if [[ -d "${save_dir}" ]] && \
               { [[ -f "${save_dir}/test_metrics.txt" ]] || [[ -f "${save_dir}/val_metrics.txt" ]]; }; then
                n_skip=$((n_skip + 1))
                continue
            fi
            printf '%s\t%s\t%s\n' "${K}" "${TASK}" "${MODE}" >> "${QUEUE_FILE}"
            n_total=$((n_total + 1))
        done
    done
done

echo "[info] queued ${n_total} jobs (${n_skip} already-done skipped)"
echo "[info] queue file: ${QUEUE_FILE}"
echo

if [[ ${n_total} -eq 0 ]]; then
    echo "[done] nothing to run."
    exit 0
fi

# ── Worker function: runs forever, pops one job at a time, returns when queue empty ──
# We define it here as a heredoc-written script so each GPU worker is a clean
# subshell with its own CUDA_VISIBLE_DEVICES.
WORKER_SCRIPT="${QUEUE_DIR}/worker.sh"
cat > "${WORKER_SCRIPT}" <<'WORKER_EOF'
#!/usr/bin/env bash
# Worker: pop one job at a time from the shared queue, run it on $CUDA_VISIBLE_DEVICES.
set -uo pipefail

GPU_ID="$1"
QUEUE_FILE="$2"
LOCK_FILE="$3"
OUT_ROOT="$4"
EPOCHS="$5"
LR="$6"
CKPT_BASENAME="$7"
ECG_FM_HB_REPO="$8"
BENCHMARK_REPO="$9"
TS="${10}"
LOG_DIR="${11}"

LOG_FILE="${LOG_DIR}/gpu${GPU_ID}_${TS}.log"
echo "[gpu${GPU_ID}] worker started, log=${LOG_FILE}" | tee -a "${LOG_FILE}"

# Activate conda env once per worker.
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate hbkim

cd "${BENCHMARK_REPO}"

while true; do
    # Atomically pop the first line under flock to avoid duplicate work.
    LINE=$(flock -x "${LOCK_FILE}" bash -c '
        QF='"${QUEUE_FILE}"'
        line=$(head -n 1 "$QF")
        if [[ -z "$line" ]]; then
            echo ""
        else
            tail -n +2 "$QF" > "${QF}.tmp" && mv "${QF}.tmp" "$QF"
            echo "$line"
        fi
    ')
    if [[ -z "${LINE}" ]]; then
        echo "[gpu${GPU_ID}] queue empty, exiting" | tee -a "${LOG_FILE}"
        break
    fi
    K=$(echo "${LINE}" | cut -f1)
    TASK=$(echo "${LINE}" | cut -f2)
    MODE=$(echo "${LINE}" | cut -f3)

    tag="ecg_fm_hb_cb${K}_v5__${TASK}__${MODE}"
    save_dir="${OUT_ROOT}/${tag}"
    job_log="${OUT_ROOT}/${tag}.log"
    CKPT="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${K}_v5/${CKPT_BASENAME}"

    ts_now=$(date '+%H:%M:%S')
    echo "[gpu${GPU_ID} ${ts_now}] START ${tag}" | tee -a "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python run.py \
        --task "${TASK}" \
        --eval_mode "${MODE}" \
        --encoder_cls src.encoders.ecg_fm_hb.ECGFMHBEncoder \
        --encoder_ckpt "${CKPT}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device cuda \
        --save_dir "${save_dir}" \
        > "${job_log}" 2>&1
    rc=$?

    ts_now=$(date '+%H:%M:%S')
    if [[ ${rc} -eq 0 ]]; then
        echo "[gpu${GPU_ID} ${ts_now}] DONE  ${tag}" | tee -a "${LOG_FILE}"
    else
        echo "[gpu${GPU_ID} ${ts_now}] FAIL  ${tag} (rc=${rc}, see ${job_log})" | tee -a "${LOG_FILE}"
    fi
done
WORKER_EOF
chmod +x "${WORKER_SCRIPT}"

# ── Spawn one worker per GPU ─────────────────────────────────────────────
PIDS=()
for gpu in ${GPUS}; do
    # Note: do NOT `disown` here. `disown` removes the job from the shell's
    # table, which makes `wait` return immediately and triggers the EXIT trap
    # before workers can grab the lock file (queue.lock then disappears →
    # `flock: No such file or directory` → workers exit instantly).
    # SIGHUP is already ignored for the whole process tree because the user
    # launches this script under nohup, so disown is unnecessary.
    nohup bash "${WORKER_SCRIPT}" \
        "${gpu}" "${QUEUE_FILE}" "${LOCK_FILE}" "${OUT_ROOT}" \
        "${EPOCHS}" "${LR}" "${CKPT_BASENAME}" \
        "${ECG_FM_HB_REPO}" "${BENCHMARK_REPO}" "${TS}" "${PWD}/logs/benchmark_v5" \
        > "logs/benchmark_v5/gpu${gpu}_${TS}.boot.log" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "[launch] GPU ${gpu}  worker pid=${pid}"
done

# Aggregate stdout from each worker into a single feed.
echo
echo "[done] ${#PIDS[@]} workers running. PIDs: ${PIDS[*]}"
cat <<EOF

# Live progress (per-worker):
tail -f logs/benchmark_v5/gpu*_${TS}.log

# All-job log (chronological):
ls -tr ${OUT_ROOT}/*.log | xargs tail -n 1 | tail -50

# Live GPU utilization across all 7 cards:
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'

# Completed-tuple count:
ls -d ${OUT_ROOT}/*/ 2>/dev/null | wc -l    # of ${n_total}

# Stop everything:
pkill -f 'launch_v5_benchmark.sh|worker.sh|run.py.*ecg_fm_hb'
EOF

# Keep the launcher process alive so the trap that cleans QUEUE_DIR fires only
# after all workers have drained the queue. (Workers read the same files; if
# QUEUE_DIR is removed too early they'll fail.)
wait "${PIDS[@]}" 2>/dev/null || true
echo "[done] all workers exited at $(date)"
