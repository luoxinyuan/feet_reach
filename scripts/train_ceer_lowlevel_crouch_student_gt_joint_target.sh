#!/usr/bin/env bash
# Continue the crouch teacher with latent adaptation, then train the student.
# In both stages the student receives ground-truth joint targets directly;
# adapt_joint_module is bypassed.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
checkpoint="${CHECKPOINT:-$PWD/outputs/ceer-lowlevel-crouch-joint-target/20261030_165223/wandb/run-20261030_165250-94z85d5d/files/checkpoint_final.pt}"
[[ "$checkpoint" = /* ]] || checkpoint="$PWD/$checkpoint"
output_dir="${OUTPUT_DIR:-$PWD/outputs/$(date +%Y%m%d)-ceer-lowlevel-crouch-student-gt-joint-target/$(date +%H%M%S)}"
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"

[[ -f "$checkpoint" ]] || { echo "Missing teacher checkpoint: $checkpoint" >&2; exit 1; }
for split in train val; do
  [[ -f "$MEMPATH/crouch_lowlevel/$split/meta.json" ]] || {
    echo "Missing crouch dataset split: $MEMPATH/crouch_lowlevel/$split" >&2
    exit 1
  }
done

mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > outputs/ceer-lowlevel-crouch-joint-target/latest_student_training.txt
sha256sum "$checkpoint" > "$output_dir/teacher_checkpoint.sha256"

for phase in adapt finetune; do
  if [[ "$phase" == adapt ]]; then
    algo=ppo_adapt
    frames="${ADAPT_FRAMES:-100000000}"
  else
    algo=ppo_finetune
    frames="${STUDENT_FRAMES:-2000000000}"
  fi
  phase_dir="$output_dir/$phase"
  mkdir -p "$phase_dir"
  cat > "$phase_dir/training_plan.json" <<PLAN
{
  "checkpoint_input": "$checkpoint",
  "phase": "$phase",
  "algorithm": "$algo",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "num_envs_per_gpu": ${NUM_ENVS:-4096},
  "total_frames": $frames,
  "dataset_paths": ["passive_whole_body_full/train", "crouch_lowlevel/train"],
  "dataset_path_weights": [1.0, 0.05],
  "student_input": "policy observation + privileged latent estimate + ground-truth 145D full-body joint target",
  "joint_target_estimator": "bypassed"
}
PLAN
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpu_list[@]}" \
    scripts/train.py task=G1/G1_base "algo=$algo" algo.student_use_gt_joint_target=true \
    "task.num_envs=${NUM_ENVS:-4096}" "total_frames=$frames" \
    "checkpoint_path=$checkpoint" vecnorm=eval save_interval=50 skip_final_eval=true \
    'task.command.dataset.mem_paths=[passive_whole_body_full/train,crouch_lowlevel/train]' \
    'task.command.dataset.path_weights=[1.0,0.05]' \
    "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-lowlevel-crouch \
    "wandb.run_name=ceer-crouch-$phase-gt-target-$(basename "$output_dir")" \
    "hydra.run.dir=$phase_dir" 2>&1 | tee "$phase_dir/train.log"
  mapfile -t checkpoints < <(find "$phase_dir" -type f -name checkpoint_final.pt)
  [[ ${#checkpoints[@]} -eq 1 ]] || {
    echo "Expected one final checkpoint under $phase_dir; found ${#checkpoints[@]}" >&2
    exit 1
  }
  checkpoint="${checkpoints[0]}"
  printf '%s\n' "$checkpoint" > "$output_dir/${phase}_checkpoint.txt"
done
