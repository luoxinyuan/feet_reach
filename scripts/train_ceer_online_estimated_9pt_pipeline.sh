#!/usr/bin/env bash
# Nine-point online-estimated compliance: GT teacher -> causal distillation -> PPO.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export TASK_NAME="${TASK_NAME:-G1/G1_ceer_online_estimated_9pt_no_ee}"
export WANDB_PROJECT="${WANDB_PROJECT:-ceer-online-estimated-9pt-compliance}"
export RUN_PREFIX="${RUN_PREFIX:-online-9pt-no-ee}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export NUM_ENVS="${NUM_ENVS:-16384}"
stamp="$(date +%Y%m%d_%H%M%S)"
export PIPELINE_DIR="${PIPELINE_DIR:-$PWD/outputs/$(date +%Y%m%d)-ceer-online-estimated-9pt-no-ee/$stamp}"
exec bash scripts/train_ceer_online_estimated_pipeline.sh "$@"
