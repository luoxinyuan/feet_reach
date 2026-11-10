#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_root="$PWD/outputs/20261104-ceer-lowlevel-force-root3-teacher-supervised-residual/full"
source_checkpoint="$PWD/outputs/20261103-ceer-lowlevel-force-root3-supervised-residual/resume_from_checkpoint_13_16384/lowlevel_finetune/wandb/run-20261103_154204-dque1v6y/files/checkpoint_final.pt"
teacher_dir="$run_root/teacher_finetune"
rollout_dir="$run_root/teacher_rollout"
residual_dir="$run_root/residual_teacher"

[[ -f "$source_checkpoint" ]] || { echo "Missing source checkpoint: $source_checkpoint" >&2; exit 2; }
for path in "$teacher_dir" "$rollout_dir" "$residual_dir"; do
  [[ ! -e "$path" ]] || { echo "Output already exists: $path" >&2; exit 2; }
done
mkdir -p "$teacher_dir" "$rollout_dir" "$residual_dir"
sha256sum "$source_checkpoint" > "$teacher_dir/source_checkpoint.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

echo "Stage 1: fine-tune CEER actor_teacher under the 3-point force task"
echo "Source checkpoint: $source_checkpoint"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES; environments per GPU: 16384"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_lowlevel_force_root3 algo=ppo_joint_target_finetune \
  algo.student_use_gt_joint_target=true algo.symmetry_augmentation=false \
  task.num_envs=16384 algo.num_minibatches=32 total_frames=2000000000 \
  checkpoint_path="$source_checkpoint" vecnorm=eval save_interval=50 skip_final_eval=true \
  wandb.mode=online wandb.project=ceer-lowlevel-force-root3 \
  wandb.run_name=ceer-lowlevel-force-root3-teacher-finetune-2B-20261104 \
  "hydra.run.dir=$teacher_dir" 2>&1 | tee "$teacher_dir/train.log"

teacher_checkpoint="$(find "$teacher_dir" -type f -name checkpoint_final.pt -print -quit)"
[[ -f "$teacher_checkpoint" ]] || { echo "No final teacher checkpoint was saved" >&2; exit 3; }
echo "$teacher_checkpoint" > "$run_root/teacher_checkpoint.txt"

echo "Stage 2: collect 4B transitions from actor_teacher"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/collect_3point_lowlevel_deltaq.py \
  --checkpoint "$teacher_checkpoint" --output-dir "$rollout_dir" \
  --total-transitions 4000000000 --num-envs 1024 --seed 20261104 --expected-crouch-weight 0.05 \
  2>&1 | tee "$rollout_dir/collect.log"

echo "Stage 3: supervised residual teacher training"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train_force_to_deltaq_residual_streaming.py \
  --data-dir "$rollout_dir" --output-dir "$residual_dir" --batch-size 16384 --seed 20261104 \
  2>&1 | tee "$residual_dir/train.log"

printf 'Pipeline complete.\nTeacher checkpoint: %s\nTeacher rollout data: %s\nResidual teacher: %s\n' \
  "$teacher_checkpoint" "$rollout_dir" "$residual_dir/residual_teacher.pt" > "$run_root/PIPELINE_COMPLETE.txt"
echo "PIPELINE_COMPLETE $run_root"
