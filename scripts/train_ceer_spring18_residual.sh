#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-spring18-residual/$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/train_ceer_residual.sh task=G1/G1_ceer_spring18_compliance wandb.project=ceer-spring18-residual-compliance "$@"
