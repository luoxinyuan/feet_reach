#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-segmented-amp/$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash scripts/train_ceer_residual.sh task=G1/G1_ceer_segmented_amp_compliance algo=ceer_residual_amp wandb.project=ceer-segmented-amp-compliance "$@"
