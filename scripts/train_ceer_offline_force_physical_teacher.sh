#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export TASK_NAME=G1/G1_ceer_offline_force_physical_teacher
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-ceer-offline-force-physical-teacher}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-offline37-budget60-physical-force-no-force-history}"
exec bash scripts/train_ceer_offline_force_prompt_teacher.sh "$@"
