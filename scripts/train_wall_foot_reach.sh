#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
phase="${1:-train}"
if (($#)); then shift; fi
case "$phase" in train|adapt|finetune) ;; *) echo 'Usage: train_wall_foot_reach.sh train|adapt|finetune [Hydra overrides]' >&2; exit 2;; esac
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$project_dir/dataset"
checkpoint_args=()
if [[ "$phase" != train ]]; then
  : "${CHECKPOINT:?Set CHECKPOINT to the previous phase checkpoint .pt file}"
  checkpoint_args+=("checkpoint_path=$CHECKPOINT")
fi
exec "${PYTHON:-python}" scripts/train.py task=G1/G1_wall_foot_reach "+exp=$phase" \
  algo.symmetry_augmentation=false "wandb.project=${WANDB_PROJECT:-wall-foot-reach}" \
  "wandb.run_name=${WANDB_RUN_NAME:-v1}" "wandb.mode=${WANDB_MODE:-disabled}" "${checkpoint_args[@]}" "$@"
