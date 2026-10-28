#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-segmented-jacobian-residual/$(date +%Y%m%d_%H%M%S)_ee-root3}"
export NUM_ENVS="${NUM_ENVS:-16384}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash scripts/train_ceer_residual.sh \
  task=G1/G1_ceer_segmented_ee_root3_compliance \
  wandb.project=ceer-segmented-jacobian-residual-compliance \
  "wandb.run_name=ee-root3-$(basename "$OUTPUT_DIR")" "$@"
