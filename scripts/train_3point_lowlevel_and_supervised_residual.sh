#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
source_checkpoint="${SOURCE_CHECKPOINT:-$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt}"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d)-ceer-lowlevel-force-root3-supervised-residual/$(date +%H%M%S)}"
lowlevel_dir="$run_root/lowlevel_finetune"
rollout_dir="$run_root/rollout"
residual_dir="$run_root/residual_teacher"
lowlevel_frames="${LOWLEVEL_FRAMES:-2000000000}"
resume_frames="${RESUME_FRAMES:-0}"
phase_frames="${PHASE_FRAMES:-$((lowlevel_frames + resume_frames))}"
[[ -f "$source_checkpoint" ]] || { echo "Missing crouch-capable GT-joint-target CEER checkpoint: $source_checkpoint" >&2; exit 2; }
[[ -f "$PWD/artifacts/ceer_segmented_jacobian/20261027-ee-root3-multi10/dataset/meta_motion.json" ]] || { echo "Missing root3 motion dataset" >&2; exit 2; }
[[ -f "$MEMPATH/crouch_lowlevel/train/meta.json" ]] || { echo "Missing crouch_lowlevel/train dataset" >&2; exit 2; }

mkdir -p "$lowlevel_dir" "$rollout_dir" "$residual_dir"
sha256sum "$source_checkpoint" > "$lowlevel_dir/source_checkpoint.sha256"
cat > "$run_root/training_plan.json" <<PLAN
{
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "stage1": "fine-tune CEER low-level with the direct full-body joint-target actor input; apply physical forces at pelvis, left hand, right hand",
  "stage1_start_checkpoint": "$source_checkpoint",
  "stage1_dataset_weights": [1.0, 0.05],
  "stage1_frames": $lowlevel_frames,
  "stage1_resume_frames": $resume_frames,
  "stage1_phase_frames": $phase_frames,
  "stage2": "roll out the fine-tuned CEER low-level and save measured force -> realized next-step full-body delta-q labels",
  "stage2_total_transitions": ${ROLLOUT_FRAMES:-4000000000},
  "stage2_dataset_weights": [1.0, 0.05],
  "stage3": "supervised force-to-delta-q training; same 512/256 LayerNorm residual trunk"
}
PLAN

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Pipeline output: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting CEER low-level fine-tuning from $source_checkpoint"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_ceer_lowlevel_force_root3 algo=ppo_finetune algo.student_use_gt_joint_target=true algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS:-16384}" "algo.num_minibatches=${NUM_MINIBATCHES:-32}" \
  "total_frames=$lowlevel_frames" "resume_frames=$resume_frames" "phase_frames=$phase_frames" \
  "checkpoint_path=$source_checkpoint" vecnorm=eval "save_interval=${SAVE_INTERVAL:-13}" skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-lowlevel-force-root3 \
  "wandb.run_name=ceer-lowlevel-force-root3-$(basename "$run_root")" "hydra.run.dir=$lowlevel_dir" \
  2>&1 | tee "$lowlevel_dir/train.log"

lowlevel_checkpoint="${LOWLEVEL_CHECKPOINT:-}"
if [[ -z "$lowlevel_checkpoint" ]]; then
  lowlevel_checkpoint="$(find "$lowlevel_dir" -type f -name checkpoint_final.pt -print -quit)"
fi
[[ -f "$lowlevel_checkpoint" ]] || { echo "Fine-tuning completed but checkpoint_final.pt was not found" >&2; exit 3; }
echo "$lowlevel_checkpoint" > "$run_root/lowlevel_checkpoint.txt"
echo "Starting distributed rollout collection: ${ROLLOUT_FRAMES:-4000000000} transitions"

torchrun --standalone --nproc_per_node="${#gpus[@]}" scripts/collect_3point_lowlevel_deltaq.py \
  --checkpoint "$lowlevel_checkpoint" --output-dir "$rollout_dir" \
  --total-transitions "${ROLLOUT_FRAMES:-4000000000}" --num-envs "${ROLLOUT_ENVS:-16384}" \
  --seed "${ROLLOUT_SEED:-20261104}" --expected-crouch-weight 0.05 \
  2>&1 | tee "$rollout_dir/collect.log"

torchrun --standalone --nproc_per_node="${#gpus[@]}" scripts/train_force_to_deltaq_residual_streaming.py \
  --data-dir "$rollout_dir" --output-dir "$residual_dir" \
  --batch-size "${SUPERVISED_BATCH_PER_GPU:-16384}" --seed "${SUPERVISED_SEED:-20261104}" \
  2>&1 | tee "$residual_dir/train.log"

printf 'Pipeline complete.\nLow-level checkpoint: %s\nRollout data: %s\nResidual teacher: %s\n' \
  "$lowlevel_checkpoint" "$rollout_dir" "$residual_dir/residual_teacher.pt" > "$run_root/PIPELINE_COMPLETE.txt"
echo "PIPELINE_COMPLETE $run_root"
