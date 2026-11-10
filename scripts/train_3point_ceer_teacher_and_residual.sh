#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
lowlevel_checkpoint="${LOWLEVEL_CHECKPOINT:-$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt}"
test -f "$lowlevel_checkpoint" || { echo "Missing crouch-capable CEER low-level checkpoint: $lowlevel_checkpoint" >&2; exit 2; }
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d)-ceer-3point-force-deltaq/$(date +%H%M%S)}"
teacher_dir="$run_root/teacher"
rollout_dir="$run_root/rollout"
residual_dir="$run_root/residual_teacher"
mkdir -p "$teacher_dir" "$rollout_dir" "$residual_dir"
if [[ -e "$teacher_dir/train.log" || -e "$rollout_dir/teacher_rollouts.npz" || -e "$residual_dir/residual_teacher.pt" ]]; then
  echo "Refusing to overwrite an existing pipeline output: $run_root" >&2; exit 2
fi
sha256sum "$lowlevel_checkpoint" > "$teacher_dir/frozen_lowlevel.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Frozen CEER low-level: $lowlevel_checkpoint"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_online_estimated_root3_crouch_tracking algo=ceer_dense_force algo.phase=teacher \
  "task.num_envs=${TEACHER_NUM_ENVS:-20480}" "total_frames=${TEACHER_FRAMES:-4000000000}" \
  "task.action.ceer_checkpoint=$lowlevel_checkpoint" checkpoint_path=null vecnorm=train save_interval=50 \
  "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-3point-force-deltaq-teacher \
  "wandb.run_name=3point-ceer-teacher-$(basename "$run_root")" "hydra.run.dir=$teacher_dir" \
  2>&1 | tee "$teacher_dir/train.log"

teacher_checkpoint="${TEACHER_CHECKPOINT:-}"
if [[ -z "$teacher_checkpoint" ]]; then
  teacher_checkpoint="$(find "$teacher_dir" -type f -name checkpoint_final.pt -print -quit)"
fi
test -f "$teacher_checkpoint" || { echo "Teacher training completed but checkpoint_final.pt was not found under $teacher_dir" >&2; exit 3; }
echo "Teacher checkpoint: $teacher_checkpoint"

"$python_bin" scripts/collect_3point_teacher_deltaq.py \
  --checkpoint "$teacher_checkpoint" --output "$rollout_dir/teacher_rollouts.npz" \
  --motion-count 160 --steps-per-motion 100 --seed 20261103 \
  2>&1 | tee "$rollout_dir/collect.log"

"$python_bin" scripts/train_force_to_deltaq_residual.py \
  --data "$rollout_dir/teacher_rollouts.npz" --output-dir "$residual_dir" --seed 20261103 \
  2>&1 | tee "$residual_dir/train.log"

cat > "$run_root/PIPELINE_COMPLETE.txt" <<EOF
Pipeline completed successfully.
Teacher checkpoint: $teacher_checkpoint
Rollout data: $rollout_dir/teacher_rollouts.npz
Residual teacher: $residual_dir/residual_teacher.pt
EOF
echo "PIPELINE_COMPLETE $run_root"
