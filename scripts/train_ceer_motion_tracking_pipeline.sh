#!/usr/bin/env bash
# Independent force-free CEER motion-tracking teacher -> adapt -> student pipeline.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="${ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE:-1}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-$project_dir/outputs/${run_stamp}-ceer-motion-tracking}"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/ceer-motion-tracking-latest.txt"

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
checkpoint_args=()
for split in train val; do
  for rel in passive_whole_body_full crouch_lowlevel; do
    [[ -f "$MEMPATH/$rel/$split/meta.json" ]] || {
      echo "Missing dataset split: $MEMPATH/$rel/$split" >&2
      exit 1
    }
  done
done

cat > "$output_dir/pipeline_plan.json" <<PLAN
{
  "pipeline": "CEER motion tracking teacher-student",
  "phases": ["train", "adapt", "finetune"],
  "algorithm": "PPOPolicy ppo_train -> ppo_adapt -> ppo_finetune",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "num_envs": ${NUM_ENVS:-4096},
  "dataset_paths": ["passive_whole_body_full/train", "crouch_lowlevel/train"],
  "dataset_path_weights": [1.0, 0.05],
  "actor_inputs": ["proprioception", "5D root-motion command", "145D full-body joint target"],
  "force_interface": "removed",
  "force_estimator": "removed; privileged-state estimator retained for student adaptation",
  "tracking_rewards": ["root position", "root rotation", "root linear velocity", "root angular velocity", "joint position", "joint velocity"],
  "ee_tracking_reward": "removed"
}
PLAN

for phase in train adapt finetune; do
  case "$phase" in
    train) frames="${TEACHER_FRAMES:-2000000000}";;
    adapt) frames="${ADAPT_FRAMES:-100000000}";;
    finetune) frames="${STUDENT_FRAMES:-2000000000}";;
  esac
  case "$phase" in
    train) algo=ppo_train; student_train=false; reward_student_train=false;;
    adapt) algo=ppo_adapt; student_train=true; reward_student_train=true;;
    finetune) algo=ppo_finetune; student_train=true; reward_student_train=true;;
  esac
  phase_dir="$output_dir/$phase"
  mkdir -p "$phase_dir"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#gpu_list[@]}" \
    scripts/train.py task=G1/G1_ceer_motion_tracking "algo=$algo" \
    "task.command.student_train=$student_train" "task.reward._student_train_=$reward_student_train" \
    algo.student_use_gt_joint_target=true algo.symmetry_augmentation=false \
    "task.num_envs=${NUM_ENVS:-4096}" "total_frames=$frames" \
    "save_interval=${SAVE_INTERVAL:-50}" skip_final_eval=true \
    "wandb.project=${WANDB_PROJECT:-ceer-motion-tracking}" "wandb.mode=${WANDB_MODE:-online}" \
    "wandb.run_name=ceer-motion-tracking-$phase-$run_stamp" \
    "hydra.run.dir=$phase_dir" "${checkpoint_args[@]}" \
    2>&1 | tee "$phase_dir/train.log"

  mapfile -t checkpoints < <(find "$phase_dir" -type f -name checkpoint_final.pt)
  [[ ${#checkpoints[@]} -eq 1 ]] || {
    echo "Expected one final checkpoint under $phase_dir; found ${#checkpoints[@]}" >&2
    exit 1
  }
  checkpoint_args=("checkpoint_path=${checkpoints[0]}")
done

printf '%s\n' "${checkpoint_args[0]#checkpoint_path=}" > "$output_dir/final_checkpoint.txt"
echo "CEER motion-tracking train/adapt/finetune pipeline finished: $output_dir"
