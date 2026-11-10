#!/usr/bin/env bash
# Online 3-point compliance, using the crouch-trained student as frozen low-level.
# Pipeline phases and budgets match the established estimated-root3 workflow.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export TASK_NAME="${TASK_NAME:-G1/G1_ceer_online_estimated_root3_student_no_ee}"
export WANDB_PROJECT="${WANDB_PROJECT:-ceer-online-estimated-root3-student-no-ee}"
export RUN_PREFIX="${RUN_PREFIX:-online-root3-student-no-ee}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export NUM_ENVS="${NUM_ENVS:-2048}"
export LOWLEVEL_CHECKPOINT="${LOWLEVEL_CHECKPOINT:-$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt}"
stamp="$(date +%H%M%S)"
export PIPELINE_DIR="${PIPELINE_DIR:-$PWD/outputs/$(date +%Y%m%d)-ceer-online-estimated-root3-student-no-ee/$stamp}"
exec bash scripts/train_ceer_online_estimated_pipeline.sh "$@"
