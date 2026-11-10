#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_root="$PWD/outputs/20261103-ceer-lowlevel-force-root3-supervised-residual/full"
resume_checkpoint="$run_root/../resume_from_checkpoint_850_16384/lowlevel_finetune/wandb/run-20261103_134015-v0yb8lth/files/checkpoint_13.pt"
resume_frames=367656960
phase_frames=2000000000
remaining_frames=$((phase_frames - resume_frames))
resume_dir="$run_root/../resume_from_checkpoint_13_16384/lowlevel_finetune"
rollout_dir="$run_root/rollout"
residual_dir="$run_root/residual_teacher"
[[ -f "$resume_checkpoint" ]] || { echo "Missing resume checkpoint: $resume_checkpoint" >&2; exit 2; }
[[ ! -f "$resume_dir/train.log" ]] || { echo "Resume output already used: $resume_dir" >&2; exit 2; }
[[ -z "$(find "$rollout_dir" -maxdepth 1 -type f -print -quit)" ]] || { echo "Rollout output is not empty: $rollout_dir" >&2; exit 2; }
[[ -z "$(find "$residual_dir" -maxdepth 1 -type f -print -quit)" ]] || { echo "Residual output is not empty: $residual_dir" >&2; exit 2; }
mkdir -p "$resume_dir" "$rollout_dir" "$residual_dir"
sha256sum "$resume_checkpoint" > "$resume_dir/resume_checkpoint.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Resuming low-level fine-tuning from $resume_checkpoint"
echo "Completed phase frames before interruption: $resume_frames / $phase_frames"
echo "Remaining low-level frames: $remaining_frames"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES; environments per GPU: 16384"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_lowlevel_force_root3 algo=ppo_finetune algo.student_use_gt_joint_target=true algo.symmetry_augmentation=false \
  task.num_envs=16384 algo.num_minibatches=32 \
  "total_frames=$remaining_frames" "resume_frames=$resume_frames" "phase_frames=$phase_frames" \
  checkpoint_path="$resume_checkpoint" vecnorm=eval save_interval=13 skip_final_eval=true \
  wandb.mode=online wandb.project=ceer-lowlevel-force-root3 \
  wandb.run_name=ceer-lowlevel-force-root3-resume-from-13-16384 \
  "hydra.run.dir=$resume_dir" 2>&1 | tee "$resume_dir/train.log"
lowlevel_checkpoint="$(find "$resume_dir" -type f -name checkpoint_final.pt -print -quit)"
[[ -f "$lowlevel_checkpoint" ]] || { echo "No final low-level checkpoint was saved" >&2; exit 3; }
echo "$lowlevel_checkpoint" > "$run_root/lowlevel_checkpoint.txt"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/collect_3point_lowlevel_deltaq.py \
  --checkpoint "$lowlevel_checkpoint" --output-dir "$rollout_dir" \
  --total-transitions 4000000000 --num-envs 16384 --seed 20261104 --expected-crouch-weight 0.05 \
  2>&1 | tee "$rollout_dir/collect.log"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train_force_to_deltaq_residual_streaming.py \
  --data-dir "$rollout_dir" --output-dir "$residual_dir" --batch-size 16384 --seed 20261104 \
  2>&1 | tee "$residual_dir/train.log"
printf 'Pipeline complete.\nLow-level checkpoint: %s\nRollout data: %s\nResidual teacher: %s\n' \
  "$lowlevel_checkpoint" "$rollout_dir" "$residual_dir/residual_teacher.pt" > "$run_root/PIPELINE_COMPLETE.txt"
echo "PIPELINE_COMPLETE $run_root"
