#!/bin/bash
# scripts/run_tokenizer.sh
# Phase 1: VQ-VAE Beat Tokenizer 학습
#
# Usage:
#   bash scripts/run_tokenizer.sh
#   bash scripts/run_tokenizer.sh --config configs/tokenizer/vqvae_base.yaml

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hbkim

CONFIG=${1:-configs/tokenizer/vqvae_base.yaml}

echo "========================================"
echo "  ECG-FM  |  Phase 1: Beat Tokenizer"
echo "  Config : $CONFIG"
echo "  Time   : $(date)"
echo "========================================"

# 단일 GPU
python -m training.tokenizer.train --config "$CONFIG"

# 멀티 GPU (DDP) 사용 시 아래 주석 해제
# torchrun --nproc_per_node=4 \
#   -m training.tokenizer.train --config "$CONFIG"
