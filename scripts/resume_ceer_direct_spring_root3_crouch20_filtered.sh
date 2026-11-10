#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_root="${OUTPUT_ROOT:-$PWD/outputs/20261105-ceer-direct-spring-root3-crouch20-filtered-v2-resume-step300}"
train_dir="$run_root/train"
source_root="$PWD/outputs/20261105-ceer-direct-spring-root3-crouch20-filtered-v2/train/wandb/run-20261105_000643-cyfa0ajj/files"
resume_checkpoint="$source_root/checkpoint_300.pt"
lowlevel_checkpoint="$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt"

# checkpoint_300 was saved after 301 batches: 591,790,080 global frames.
phase_frames=4000000000
resume_frames=591790080
remaining_frames=$((phase_frames - resume_frames))

test -f "$resume_checkpoint" || { echo "Missing resume checkpoint: $resume_checkpoint" >&2; exit 2; }
test -f "$lowlevel_checkpoint" || { echo "Missing frozen low-level checkpoint: $lowlevel_checkpoint" >&2; exit 2; }
test ! -f "$train_dir/train.log" || { echo "Resume output already used: $train_dir" >&2; exit 2; }
mkdir -p "$train_dir"
sha256sum "$resume_checkpoint" > "$train_dir/resume_checkpoint.sha256"
sha256sum "$lowlevel_checkpoint" > "$train_dir/frozen_lowlevel.sha256"
sha256sum artifacts/compliance_dataset_screening/20261104_dynamic_trajectory_screening/root3_allowlist.json artifacts/compliance_dataset_screening/20261104_dynamic_trajectory_screening/crouch_allowlist.json > "$train_dir/dataset_allowlists.sha256"

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Run root: $run_root"
echo "Resume checkpoint: $resume_checkpoint"
echo "Completed global frames: $resume_frames / $phase_frames"
echo "Remaining global frames: $remaining_frames"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: 20480"
echo "Force method: direct root/hand next-frame springs, gain 80 N/m, 30 N/point and 60 N total caps"
echo "Force history offsets: [0, 1, 2, 4, 8, 16]"
echo "Dataset: screened root3 + crouch allowlists, 80/20 source weights"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_direct_spring_root3_crouch20_force_history_filtered \
  algo=ceer_dense_force algo.phase=teacher \
  task.num_envs=20480 "total_frames=$remaining_frames" "resume_frames=$resume_frames" "phase_frames=$phase_frames" \
  "task.action.ceer_checkpoint=$lowlevel_checkpoint" "checkpoint_path=$resume_checkpoint" \
  vecnorm=train save_interval=50 \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=ceer-direct-spring-root3-crouch20 \
  wandb.run_name=direct-spring-root3-crouch20-filtered-resume-step300 \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"
