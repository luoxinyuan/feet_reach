#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
phase="${1:?Usage: train_ceer_online_estimated_stage.sh teacher|distill|finetune [hydra overrides]}"; shift
case "$phase" in teacher|distill|finetune) ;; *) echo "Unknown phase: $phase" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
checkpoint="${CHECKPOINT:-null}"
if [[ "$phase" != teacher && "$checkpoint" == null ]]; then echo "$phase requires CHECKPOINT from the preceding stage" >&2; exit 2; fi
if [[ "$checkpoint" != null ]]; then checkpoint="$(realpath "$checkpoint")"; test -f "$checkpoint"; fi
norm=train; default_frames=4000000000
if [[ "$phase" != teacher ]]; then norm=eval; fi
if [[ "$phase" == distill ]]; then default_frames=1000000000; fi
output="${OUTPUT_DIR:-$PWD/outputs/ceer-online-estimated-root3/$(date +%Y%m%d_%H%M%S)/$phase}"
task_name="${TASK_NAME:-G1/G1_ceer_online_estimated_root3}"
wandb_project="${WANDB_PROJECT:-ceer-online-estimated-root3-compliance}"
run_prefix="${RUN_PREFIX:-online-root3}"
if [[ -f "$output/train.log" ]]; then echo "Output already used: $output; choose a fresh OUTPUT_DIR (and CHECKPOINT to resume)" >&2; exit 2; fi
mkdir -p "$output"
sha256sum .cache/ceer/checkpoint_final.pt > "$output/frozen_ceer.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
 "task=$task_name" algo=ceer_dense_force "algo.phase=$phase" \
 "task.num_envs=${NUM_ENVS:-2048}" "total_frames=${TOTAL_FRAMES:-$default_frames}" \
 "checkpoint_path=$checkpoint" "vecnorm=$norm" save_interval=50 \
 "wandb.mode=${WANDB_MODE:-online}" "wandb.project=$wandb_project" \
 "wandb.run_name=$run_prefix-$phase-$(basename "$(dirname "$output")")" \
 "hydra.run.dir=$output" "$@" 2>&1 | tee "$output/train.log"
