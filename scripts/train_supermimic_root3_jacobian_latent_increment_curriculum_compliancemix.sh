#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_supermimic_online_estimated_root3_crouch20_force_history_latent_curriculum_compliancemix}"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
preflight="${PREFLIGHT_JSON:-$PWD/artifacts/compliance_eval/20261109-supermimic-root3-crouch20-force-history/preflight-crouch-final/preflight.json}"
run_root="${OUTPUT_ROOT:-$PWD/outputs/20261118-supermimic-root3-jacobian-latent-increment-curriculum-compliancemix-gpu567}"
train_dir="$run_root/train"

for f in "$checkpoint" "$deploy_config" "$preflight" "$PWD/dataset/compliance_mix_29h_v1/manifest.json"; do
  [[ -f "$f" ]] || { echo "Missing required artifact: $f" >&2; exit 2; }
done
"$python_bin" - "$preflight" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))['summary']
if s['fall_clips'] != 0 or s['clip_mean_joint_mae_rad'] > 0.25:
    raise SystemExit(f'Preflight failed: {s}')
print('SuperMimic preflight passed:', s)
PY
[[ ! -f "$train_dir/train.log" ]] || { echo "Output already used: $train_dir" >&2; exit 2; }
mkdir -p "$train_dir"
cp "$checkpoint" "$train_dir/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$train_dir/supermimic_deploy_config.json"
cp "$preflight" "$train_dir/preflight.json"
cp dataset/compliance_mix_29h_v1/manifest.json "$train_dir/dataset_manifest.json"
cp cfg/task/G1/G1_supermimic_online_estimated_root3_crouch20_force_history_latent_curriculum_compliancemix.yaml "$train_dir/task_config.yaml"
cp "$0" "$train_dir/launch.sh"
sha256sum "$checkpoint" "$deploy_config" dataset/compliance_mix_29h_v1/manifest.json > "$train_dir/input_sha256.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "Run root: $run_root"
  echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
  echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
  echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
  echo "Task config: $task_config"
  echo "Curriculum: episode-start seed latent; smoothstep future-GT-base removal over progress 0.15..0.80"
  echo "Action: 32-D tanh latent increment accumulated at scale 0.025"
  echo "Force: online root + left/right hand Jacobian; history [0,1,2,4,8,16]"
  echo "Dataset: ComplianceMix-29h-v1"
  echo "Frozen SuperMimic: $checkpoint"
} | tee "$train_dir/run_info.txt"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" \
  +compliance_data=compliance_mix_29h_v1 \
  algo=ceer_dense_force algo.phase=teacher +algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "task.action.latent_increment_scale=${LATENT_INCREMENT_SCALE:-0.025}" \
  "total_frames=${TOTAL_FRAMES:-4000000000}" checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=supermimic-online-root3-crouch20-force-history \
  "wandb.run_name=supermimic-root3-jacobian-latent-increment-curriculum-compliancemix-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"
find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
