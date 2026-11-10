#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export TASK_NAME=G1/G1_ceer_gt_delta_q_teacher ALGO_NAME=ceer_gt_delta_q_teacher
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-ceer-gt-delta-q-teacher}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-gt-reference-delta-q-no-physical-force-no-force-history}"
exec bash scripts/train_ceer_offline_force_prompt_teacher.sh "$@"
