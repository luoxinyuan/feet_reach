#!/usr/bin/env bash
# Run one student phase from the preceding DenseForcePPO checkpoint.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
phase="${1:?Usage: train_supermimic_force_estimator_student_stage.sh distill|finetune [Hydra overrides]}"; shift
case "$phase" in distill|finetune) ;; *) echo "Student phase must be distill or finetune" >&2; exit 2;; esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export SUPER_MIMIC_ROOT="${SUPER_MIMIC_ROOT:-/home/xl521/supermimic_force_transition}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the preceding-stage or teacher checkpoint}"
checkpoint="$(realpath "$checkpoint")"
test -f "$checkpoint"
lowlevel_ckpt="$SUPER_MIMIC_ROOT/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="$SUPER_MIMIC_ROOT/pretrained/e11c_0220000/deploy_config.json"
test -f "$lowlevel_ckpt" && test -f "$deploy_config"

if [[ "$phase" == distill ]]; then
  default_frames="${TOTAL_FRAMES:-${DISTILL_FRAMES:-1000000000}}"
else
  default_frames="${TOTAL_FRAMES:-${FINETUNE_FRAMES:-4000000000}}"
fi
output="${OUTPUT_DIR:-$PWD/outputs/supermimic-9pt-force-estimator-student/$(date +%Y%m%d_%H%M%S)/$phase}"
if [[ -e "$output/train.log" ]]; then echo "Output already used: $output" >&2; exit 2; fi
mkdir -p "$output"
cp "$lowlevel_ckpt" "$output/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$output/supermimic_deploy_config.json"
sha256sum "$checkpoint" "$lowlevel_ckpt" "$deploy_config" > "$output/input_sha256.txt"
printf '%s\n' "$checkpoint" > "$output/input_checkpoint.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

echo "Phase: $phase"
echo "Output: $output"
echo "Checkpoint: $checkpoint"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Estimator history: 17 frames x 93 features; output: 9 points x 6 offsets x 3 force axes"
echo "Transitions: $default_frames"
echo "SuperMimic root: $SUPER_MIMIC_ROOT"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  task=G1/G1_supermimic_online_estimated_9pt_crouch20_force_history_student \
  algo=supermimic_force_estimator_student "algo.phase=$phase" \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=$default_frames" \
  "checkpoint_path=$checkpoint" vecnorm=eval save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  "wandb.project=supermimic-online-estimated-9pt-force-estimator-student" \
  "wandb.run_name=supermimic-9pt-force-estimator-$phase-$(basename "$(dirname "$output")")" \
  "hydra.run.dir=$output" "$@" 2>&1 | tee "$output/train.log"
find "$output" -type f -name 'checkpoint_final.pt' -print -quit > "$output/final_checkpoint.txt"
echo "Phase finished: $output"
