#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export NUM_ENVS="${NUM_ENVS:-20480}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-jacobian-no-ee-residual/$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec bash scripts/train_ceer_residual.sh \
  task=G1/G1_ceer_jacobian_no_ee_compliance \
  wandb.project=ceer-jacobian-no-ee-residual-compliance \
  "wandb.run_name=jacobian9-no-ee-$(basename "$OUTPUT_DIR")" "$@"
