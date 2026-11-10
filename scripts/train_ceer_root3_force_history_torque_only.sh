#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
lowlevel_checkpoint="${LOWLEVEL_CHECKPOINT:-$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt}"
test -f "$lowlevel_checkpoint" || { echo "Missing crouch-trained CEER low-level checkpoint: $lowlevel_checkpoint" >&2; exit 2; }
run_root="${OUTPUT_ROOT:-$PWD/outputs/20261104-ceer-online-root3-crouch20-force-history-torque-only}"
train_dir="$run_root/train"
mkdir -p "$train_dir"
if [[ -f "$train_dir/train.log" ]]; then echo "Output already used: $train_dir" >&2; exit 2; fi
sha256sum "$lowlevel_checkpoint" > "$train_dir/frozen_lowlevel.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-20480}"
echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
echo "Force history offsets: [0, 1, 2, 4, 8, 16]"
echo "Crouch dataset path weights: [0.8, 0.2]"
echo "Force objective: torque and joint motion only; net-force matching disabled"
echo "Frozen crouch-trained CEER low-level: $lowlevel_checkpoint"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_online_estimated_root3_student_no_ee_crouch20_force_history_torque_only \
  algo=ceer_dense_force algo.phase=teacher \
  "task.num_envs=${NUM_ENVS_PER_GPU:-20480}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  "task.action.ceer_checkpoint=$lowlevel_checkpoint" checkpoint_path=null vecnorm=train save_interval=50 \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=ceer-online-estimated-root3-student-no-ee \
  "wandb.run_name=online-root3-no-ee-crouch20-force-history-torque-only-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"
