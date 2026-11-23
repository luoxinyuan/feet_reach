#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_supermimic_surface_observable3_direct_latent_compliancemix}"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d)-supermimic-surface3-direct-latent-compliancemix/$(date +%H%M%S)}"
train_dir="$run_root/train"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
if [[ ! -f "$checkpoint" || ! -f "$deploy_config" ]]; then
  echo "Missing frozen SuperMimic checkpoint or deploy config." >&2
  exit 2
fi
if [[ -f "$train_dir/train.log" ]]; then
  echo "Output already used: $train_dir" >&2
  exit 2
fi

mkdir -p "$train_dir"
cp "$checkpoint" "$train_dir/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$train_dir/supermimic_deploy_config.json"
sha256sum "$checkpoint" "$deploy_config" > "$train_dir/supermimic_sha256.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
echo "Task config: $task_config"
echo "Force interface: up to 3 motion-observable surface contacts over the whole body"
echo "Force history offsets: [0, 1, 2, 4, 8, 16]"
echo "Dataset: ComplianceMix-29h-v1, same weights as direct-latent 3KP baseline"
echo "PPO action: direct 32-D SuperMimic latent, no GT motion latent"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" \
  +compliance_data=compliance_mix_29h_v1 \
  algo=ceer_force_prompt_teacher +algo.symmetry_augmentation=false \
  'algo.in_keys=[policy,force_gt_]' \
  'algo.critic_in_keys=[policy,priv,joint_target,force_gt_]' \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true seed=0 \
  '~task.command.joint_delta_gain' '~task.command.joint_feedback_gain' \
  '~task.command.force_penalty' '~task.command.smooth_penalty' \
  '~task.command.resample_attempts' '~task.command.force_points' \
  '~task.command.estimator_history' '~task.command.force_prompt_history_offsets' \
  '~task.observation.force_gt_.dense_force_gt_history' \
  '~task.observation.estimator_history_' \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=supermimic-surface3-direct-latent-compliancemix \
  "wandb.run_name=supermimic-surface3-direct-latent-compliancemix-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"

find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
