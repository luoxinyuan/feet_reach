#!/usr/bin/env bash
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
checkpoint="$PWD/.cache/ceer/checkpoint_final.pt"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-$PWD/outputs/ceer-lowlevel-crouch-joint-target/$stamp}"
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"

[[ -f "$checkpoint" ]] || { echo "Missing CEER initialization checkpoint: $checkpoint" >&2; exit 1; }
for split in train val; do
  [[ -f "$MEMPATH/crouch_lowlevel/$split/meta.json" ]] || {
    echo "Missing crouch dataset split: $MEMPATH/crouch_lowlevel/$split" >&2
    exit 1
  }
done

mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > outputs/ceer-lowlevel-crouch-joint-target/latest_training.txt
sha256sum "$checkpoint" > "$output_dir/initial_ceer_checkpoint.sha256"
cat > "$output_dir/training_plan.json" <<EOF
{
  "initial_checkpoint": "$checkpoint",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "num_envs_per_gpu": ${NUM_ENVS:-4096},
  "total_frames": ${TOTAL_FRAMES:-2000000000},
  "dataset_paths": ["passive_whole_body_full/train", "crouch_lowlevel/train"],
  "dataset_path_weights": [1.0, 0.05],
  "crouch_sampling_probability_per_reset": 0.0476190476,
  "algorithm": "ppo_joint_target_finetune",
  "actor_input": "policy observation + privileged state encoding + direct full-body joint target",
  "joint_target_estimators_trained": false,
  "final_evaluation_skipped_to_avoid_training_exit_oom": true
}
EOF

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpu_list[@]}" \
  scripts/train.py task=G1/G1_base algo=ppo_joint_target_finetune \
  "task.num_envs=${NUM_ENVS:-4096}" "total_frames=${TOTAL_FRAMES:-2000000000}" \
  "checkpoint_path=$checkpoint" vecnorm=eval save_interval=50 skip_final_eval=true \
  'task.command.dataset.mem_paths=[passive_whole_body_full/train,crouch_lowlevel/train]' \
  'task.command.dataset.path_weights=[1.0,0.05]' \
  "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-lowlevel-crouch \
  "wandb.run_name=ceer-crouch-joint-target-$stamp" \
  "hydra.run.dir=$output_dir" 2>&1 | tee "$output_dir/train.log"
