#!/usr/bin/env bash
# Run downstream benchmark finetune for HEEDB-pretrained ECG-FM on the
# tyoon11/benchmark repo with paper-canonical conditions.
#
# Targets the v4 codebook ablation: cb128 / cb256 / cb512 / cb1024 / cb2048.
# Defaults match the paper (lr=1e-3, epochs=100, finetune_linear). Override
# via env vars; see the block below.
#
# Usage:
#   bash scripts/run_benchmark_finetune.sh                 # all sizes × all tasks
#   SIZES="256 1024" TASKS="ptbxl_super code15" \
#       bash scripts/run_benchmark_finetune.sh
#   SIZES=256 TASKS=ptbxl_super EVAL_MODES=linear_probe EPOCHS=10 \
#       bash scripts/run_benchmark_finetune.sh             # smoke
#
# Required env (auto-defaulted for this server):
#   ECG_DATA_ROOT     : data root containing h5/physionet, h5/code15, ...
#   ECG_CKPT_ROOT     : root for OTHER FM checkpoints (paper baselines)
#   ECG_FM_HB_REPO    : ecg-fm pretrain repo (where models/, data/preprocessing
#                       live so the adapter can import them)
#
# This script does NOT compete for resources with paper baselines — it only
# runs our HEEDB sizes. To get baselines, run benchmark/run_full_benchmark.sh
# separately with the default model list.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────
ECG_FM_HB_REPO="${ECG_FM_HB_REPO:-/home/irteam/local-node-d/hbkimi/ecg-fm}"
BENCHMARK_REPO="${BENCHMARK_REPO:-/home/irteam/local-node-d/hbkimi/benchmark}"
export ECG_FM_HB_REPO
export ECG_DATA_ROOT="${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}"

# ── Knobs (override via env) ─────────────────────────────────────────────
# Codebook sizes to evaluate. cb128 is still pre-training (epoch 10 of 200);
# include or exclude as needed.
SIZES="${SIZES:-128 256 512 1024 2048}"

# Tasks. Paper canonical 17 by default; override with TASKS=... to subset.
TASKS="${TASKS:-ptbxl_super ptbxl_sub ptbxl_diag ptbxl_form ptbxl_rhythm ptbxl_all chapman chapman_rhythm cpsc2018 cpsc_extra ningbo georgia ptb code15 sph_diag zzu_pecg echonext}"

# Eval modes. Paper reports four; default to finetune_linear (the request).
# Add 'linear_probe' to also include frozen probing under the same conditions.
EVAL_MODES="${EVAL_MODES:-finetune_linear}"

# Hyperparams (paper defaults, override only for smoke tests).
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-}"   # empty → use task yaml default
DEVICE="${DEVICE:-cuda}"

# Pretrain checkpoint pattern: best.pt by default; override to last.pt for cb128.
CKPT_BASENAME="${CKPT_BASENAME:-best.pt}"

# Pretrain version: v4 (default) or v5 (MoRyECG arch). The encoder adapter
# auto-routes via model_cfg["arch"], so the only thing that changes here is
# the checkpoint directory: pretrain_heedb_cb${K}_${VERSION}.
VERSION="${VERSION:-v4}"

# Output root — one timestamp shared across all (size, task, mode) tuples so
# the auto-generated results_all.csv aggregates them.
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_REPO}/results/${TIMESTAMP}_ecg_fm_hb}"
mkdir -p "${OUT_ROOT}"

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -d "${BENCHMARK_REPO}" ]]; then
    echo "[error] BENCHMARK_REPO not found: ${BENCHMARK_REPO}" >&2
    exit 2
fi
if [[ ! -d "${ECG_FM_HB_REPO}/models" ]]; then
    echo "[error] ECG_FM_HB_REPO not found or missing models/: ${ECG_FM_HB_REPO}" >&2
    exit 2
fi

# Activate conda env (hbkim per CLAUDE.md memory).
# Always source conda.sh — under nohup the conda binary may be on PATH but the
# shell function for `conda activate` is only defined after sourcing conda.sh.
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "[error] could not locate conda.sh — install or set PATH" >&2
    exit 3
fi
conda activate hbkim

cd "${BENCHMARK_REPO}"

# ── Run loop ─────────────────────────────────────────────────────────────
fail=0
for K in ${SIZES}; do
    CKPT="${ECG_FM_HB_REPO}/checkpoints/pretrain_heedb_cb${K}_${VERSION}/${CKPT_BASENAME}"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[skip] cb${K}_${VERSION}: checkpoint not found at ${CKPT}"
        continue
    fi
    for TASK in ${TASKS}; do
        for MODE in ${EVAL_MODES}; do
            tag="ecg_fm_hb_cb${K}_${VERSION}__${TASK}__${MODE}"
            save_dir="${OUT_ROOT}/${tag}"
            log_file="${OUT_ROOT}/${tag}.log"
            if [[ -d "${save_dir}" ]] && [[ -f "${save_dir}/test_metrics.txt" || -f "${save_dir}/val_metrics.txt" ]]; then
                echo "[skip] ${tag} already has results — delete ${save_dir} to redo"
                continue
            fi
            echo "[run]  ${tag}  ckpt=${CKPT}"
            extra_args=()
            if [[ -n "${BATCH_SIZE}" ]]; then
                extra_args+=(--batch_size "${BATCH_SIZE}")
            fi
            python run.py \
                --task "${TASK}" \
                --eval_mode "${MODE}" \
                --encoder_cls src.encoders.ecg_fm_hb.ECGFMHBEncoder \
                --encoder_ckpt "${CKPT}" \
                --epochs "${EPOCHS}" \
                --lr "${LR}" \
                --device "${DEVICE}" \
                --save_dir "${save_dir}" \
                "${extra_args[@]}" \
                2>&1 | tee "${log_file}" || {
                    echo "[fail] ${tag} (see ${log_file})"
                    fail=$((fail + 1))
                    continue
                }
        done
    done
done

echo
echo "[done] results root: ${OUT_ROOT}"
echo "[done] aggregated CSV: ${OUT_ROOT}/results_all.csv (if any run completed)"
if [[ ${fail} -gt 0 ]]; then
    echo "[warn] ${fail} run(s) failed — see per-tag .log files"
    exit 1
fi
