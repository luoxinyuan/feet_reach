#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_supermimic_online_estimated_root3_crouch20_force_history_latent_curriculum_combined_slow}"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
preflight="${PREFLIGHT_JSON:-$PWD/artifacts/compliance_eval/20261109-supermimic-root3-crouch20-force-history/preflight-crouch-final/preflight.json}"
new_manifest="$PWD/dataset/compliance_mix_29h_v1_legacy27/manifest.json"
new_source_manifest="$PWD/dataset/compliance_mix_29h_v1/manifest.json"
legacy_manifest="$PWD/artifacts/ceer_segmented_jacobian/20261027-ee-root3-multi10/manifest.json"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d)-supermimic-root3-jacobian-latent-increment-curriculum-combined-16b-allgpu}"
train_dir="$run_root/train"

for f in "$checkpoint" "$deploy_config" "$preflight" "$new_manifest" "$new_source_manifest" "$legacy_manifest"; do
  [[ -f "$f" ]] || { echo "Missing required artifact: $f" >&2; exit 2; }
done
[[ -d "$PWD/dataset/crouch_lowlevel/train" ]] || { echo "Missing legacy crouch dataset" >&2; exit 2; }
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
cp "$new_manifest" "$train_dir/compliance_mix_29h_v1_legacy27_manifest.json"
cp "$new_source_manifest" "$train_dir/compliance_mix_29h_v1_source_manifest.json"
cp "$legacy_manifest" "$train_dir/legacy_root3_manifest.json"
cp cfg/compliance_data/compliance_mix_29h_plus_legacy_root3.yaml "$train_dir/dataset_mix.yaml"
cp "cfg/task/G1/${task_config##*/}.yaml" "$train_dir/task_config.yaml"
cp "$0" "$train_dir/launch.sh"
sha256sum "$checkpoint" "$deploy_config" "$new_manifest" "$new_source_manifest" "$legacy_manifest" > "$train_dir/input_sha256.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "Run root: $run_root"
  echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
  echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
  echo "Total transitions: ${TOTAL_FRAMES:-16000000000}"
  echo "Task config: $task_config"
  echo "Dataset mix: 50% ComplianceMix-29h-v1 projected to the common legacy 27-body schema, 50% legacy root3+crouch (legacy internal ratio 1:0.05)"
  echo "Curriculum: smoothstep begins at 5% and reaches fully autonomous latent increments at 85% of the 16B-frame run"
  echo "Action: 32-D tanh latent increment, accumulated at scale ${LATENT_INCREMENT_SCALE:-0.025}"
  echo "Force: online 3KP Jacobian (root + left/right hand); history [0,1,2,4,8,16]"
  echo "Frozen SuperMimic: $checkpoint"
} | tee "$train_dir/run_info.txt"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" \
  algo=ceer_dense_force algo.phase=teacher +algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "task.action.latent_increment_scale=${LATENT_INCREMENT_SCALE:-0.025}" \
  "total_frames=${TOTAL_FRAMES:-16000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=supermimic-online-root3-crouch20-force-history \
  "wandb.run_name=supermimic-root3-jacobian-latent-increment-curriculum-combined-16b-$(date +%Y%m%d-%H%M%S)" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"

find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
