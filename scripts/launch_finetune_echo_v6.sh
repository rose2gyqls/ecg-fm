#!/usr/bin/env bash
# Downstream finetune launcher for two job sets, drained through a single
# 7-GPU flock-based queue (one job per GPU at a time):
#
#   Set A — v4 codebook ablation × echonext only:
#     {cb128, cb256, cb512, cb1024, cb2048} × {echonext} × {linear_probe,
#     attention_probe, finetune_linear}  =  15 jobs
#
#   Set B — v6 cb1024 × all 17 tasks:
#     {cb1024_v6} × {ptbxl_super, ptbxl_sub, ptbxl_diag, ptbxl_form,
#     ptbxl_rhythm, ptbxl_all, chapman, chapman_rhythm, cpsc2018, cpsc_extra,
#     ningbo, georgia, ptb, code15, sph_diag, zzu_pecg, echonext} × 3 modes
#     =  51 jobs
#
# Total: 66 (size, task, mode, version) tuples.
#
# v6 reuses the v4 tokenizer (per moryecg_heedb_cb1024_v6.yaml) and v4
# preprocessing (PREPROC_VERSION="v4" in src/encoders/ecg_fm_hb.py), so the
# v4 cache is shared across both sets — no re-precompute.
#
# Usage:
#   nohup bash scripts/launch_finetune_echo_v6.sh > finetune_echo_v6.log 2>&1 &
#
# Override:
#   GPUS="0 1 2 3 4 5 6"  bash scripts/launch_finetune_echo_v6.sh
#   EVAL_MODES=finetune_linear  bash scripts/launch_finetune_echo_v6.sh   # smoke
#   EPOCHS=10  bash scripts/launch_finetune_echo_v6.sh                    # smoke
#
# Stop:
#   pkill -f 'launch_finetune_echo_v6.sh|run.py.*ecg_fm_hb'
#
# Inspect:
#   tail -f logs/benchmark_finetune_echo_v6/gpu*_<TS>.log
#   ls -d <OUT_ROOT>/*/  | wc -l

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/benchmark_finetune_echo_v6

TS=$(date +%Y%m%d_%H%M%S)
QUEUE_DIR=$(mktemp -d -t ecgfm_finetune_echo_v6.XXXXXX)
trap 'rm -rf "${QUEUE_DIR}"' EXIT

# ── Required env (defaults work on this server) ──────────────────────────
export ECG_FM_HB_CACHE="${ECG_FM_HB_CACHE:-/home1/irteam/local-node-d/hbkimi/.cache/ecg_fm_hb_v4}"
export ECG_DATA_ROOT="${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}"
export ECG_FM_HB_REPO="${ECG_FM_HB_REPO:-$(pwd)}"

# ── Knobs ────────────────────────────────────────────────────────────────
# Set A: v4 ablation sizes, only on echonext.
V4_SIZES="${V4_SIZES:-128 256 512 1024 2048}"
V4_TASKS="${V4_TASKS:-echonext}"

# Set B: v6 cb1024, all 17 paper-canonical tasks (includes echonext).
V6_SIZES="${V6_SIZES:-1024}"
V6_TASKS_DEFAULT="ptbxl_super ptbxl_sub ptbxl_diag ptbxl_form ptbxl_rhythm ptbxl_all chapman chapman_rhythm cpsc2018 cpsc_extra ningbo georgia ptb code15 sph_diag zzu_pecg echonext"
V6_TASKS="${V6_TASKS:-${V6_TASKS_DEFAULT}}"

EVAL_MODES="${EVAL_MODES:-linear_probe attention_probe finetune_linear}"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
CKPT_BASENAME="${CKPT_BASENAME:-best.pt}"
BENCHMARK_REPO="${BENCHMARK_REPO:-/home/irteam/local-node-d/hbkimi/benchmark}"

# All H200s on this host. Override GPUS=... to use a subset.
GPUS="${GPUS:-0 1 2 3 4 5 6}"

# Single shared OUT_ROOT so the benchmark's CSV aggregator finds every tuple.
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_REPO}/results/${TS}_ecg_fm_hb_echo_v6}"
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
echo "[info] set A v4  : sizes=${V4_SIZES}  tasks=${V4_TASKS}"
echo "[info] set B v6  : sizes=${V6_SIZES}  tasks=${V6_TASKS}"
echo "[info] modes     : ${EVAL_MODES}"
echo "[info] epochs    : ${EPOCHS}   lr=${LR}"
echo "[info] cache     : ${ECG_FM_HB_CACHE} (${n_cache} files)"
echo "[info] out_root  : ${OUT_ROOT}"
echo

# ── Build job queue: one line per (size, task, mode, version) tuple ──────
# Set A (v4 × echo) enqueued first so it drains before set B starts pulling
# at the same priority. Workers cooperatively dequeue, so v4 echo finishes
# ~first while later v4 jobs and v6 jobs interleave naturally.
QUEUE_FILE="${QUEUE_DIR}/jobs.txt"
LOCK_FILE="${QUEUE_DIR}/queue.lock"
: > "${QUEUE_FILE}"
: > "${LOCK_FILE}"

enqueue_set() {
    local version="$1" sizes="$2" tasks="$3"
    local n_added=0 n_skip=0 n_missing=0
    for K in ${sizes}; do
        local CKPT="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${K}_${version}/${CKPT_BASENAME}"
        if [[ ! -f "${CKPT}" ]]; then
            echo "[skip-size] cb${K}_${version}: checkpoint missing at ${CKPT}"
            n_missing=$((n_missing + 1))
            continue
        fi
        for TASK in ${tasks}; do
            for MODE in ${EVAL_MODES}; do
                local tag="ecg_fm_hb_cb${K}_${version}__${TASK}__${MODE}"
                local save_dir="${OUT_ROOT}/${tag}"
                if [[ -d "${save_dir}" ]] && \
                   { [[ -f "${save_dir}/test_metrics.txt" ]] || [[ -f "${save_dir}/val_metrics.txt" ]]; }; then
                    n_skip=$((n_skip + 1))
                    continue
                fi
                printf '%s\t%s\t%s\t%s\n' "${K}" "${TASK}" "${MODE}" "${version}" >> "${QUEUE_FILE}"
                n_added=$((n_added + 1))
            done
        done
    done
    echo "[enqueue ${version}] +${n_added} jobs (${n_skip} done-skipped, ${n_missing} missing-ckpt)"
}

enqueue_set "v4" "${V4_SIZES}" "${V4_TASKS}"
enqueue_set "v6" "${V6_SIZES}" "${V6_TASKS}"

n_total=$(wc -l < "${QUEUE_FILE}")
echo "[info] queued ${n_total} jobs total"
echo "[info] queue file: ${QUEUE_FILE}"
echo

if [[ ${n_total} -eq 0 ]]; then
    echo "[done] nothing to run."
    exit 0
fi

# ── Worker function: pop one job at a time, run on its pinned GPU ────────
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

# Activate conda env once per worker (CLAUDE.md: always use hbkim).
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
    VERSION=$(echo "${LINE}" | cut -f4)

    tag="ecg_fm_hb_cb${K}_${VERSION}__${TASK}__${MODE}"
    save_dir="${OUT_ROOT}/${tag}"
    job_log="${OUT_ROOT}/${tag}.log"
    CKPT="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${K}_${VERSION}/${CKPT_BASENAME}"

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
    # Note: do NOT `disown`. `disown` removes the job from the shell's table,
    # which makes `wait` return immediately and the EXIT trap fires before
    # workers can grab the lock file. nohup at outer launch handles SIGHUP.
    nohup bash "${WORKER_SCRIPT}" \
        "${gpu}" "${QUEUE_FILE}" "${LOCK_FILE}" "${OUT_ROOT}" \
        "${EPOCHS}" "${LR}" "${CKPT_BASENAME}" \
        "${ECG_FM_HB_REPO}" "${BENCHMARK_REPO}" "${TS}" "${PWD}/logs/benchmark_finetune_echo_v6" \
        > "logs/benchmark_finetune_echo_v6/gpu${gpu}_${TS}.boot.log" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "[launch] GPU ${gpu}  worker pid=${pid}"
done

echo
echo "[done] ${#PIDS[@]} workers running. PIDs: ${PIDS[*]}"
cat <<EOF

# Live progress (per-worker):
tail -f logs/benchmark_finetune_echo_v6/gpu*_${TS}.log

# All-job log (chronological, latest line each):
ls -tr ${OUT_ROOT}/*.log 2>/dev/null | xargs -r tail -n 1 | tail -50

# Live GPU utilization across all 7 cards:
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'

# Completed-tuple count:
ls -d ${OUT_ROOT}/*/ 2>/dev/null | wc -l    # of ${n_total}

# Stop everything:
pkill -f 'launch_finetune_echo_v6.sh|run.py.*ecg_fm_hb'
EOF

# Keep the launcher alive so the EXIT trap (cleans QUEUE_DIR) fires only after
# all workers drain. Workers read from QUEUE_DIR — removing it early kills them.
wait "${PIDS[@]}" 2>/dev/null || true
echo "[done] all workers exited at $(date)"
