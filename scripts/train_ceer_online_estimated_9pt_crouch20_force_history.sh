#!/usr/bin/env bash
# 9-point Online Jacobian with force history and 20% crouch data.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export TASK_NAME="${TASK_NAME:-G1/G1_ceer_online_estimated_9pt_no_ee_crouch20_force_history}"
export WANDB_PROJECT="${WANDB_PROJECT:-ceer-online-estimated-9pt-crouch20-force-history}"
export RUN_PREFIX="${RUN_PREFIX:-online-9pt-crouch20-force-history}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export NUM_ENVS="${NUM_ENVS:-16384}"
export LOWLEVEL_CHECKPOINT="${LOWLEVEL_CHECKPOINT:-$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt}"
stamp="$(date +%Y%m%d_%H%M%S)"
export PIPELINE_DIR="${PIPELINE_DIR:-$PWD/outputs/$(date +%Y%m%d)-ceer-online-estimated-9pt-crouch20-force-history/$stamp}"
exec bash scripts/train_ceer_online_estimated_pipeline.sh "$@"
