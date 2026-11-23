#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_supermimic_surface5_direct_latent_curriculum}"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-supermimic-surface5-direct-latent-curriculum-legacy}"
train_dir="$run_root/train"
for f in "$checkpoint" "$deploy_config" "$PWD/artifacts/surface_compliance/geometry" "$PWD/artifacts/ceer_segmented_jacobian/geometry.json" "$PWD/dataset/passive_whole_body_full/manifest.json" "$PWD/dataset/crouch_lowlevel/manifest.json"; do
  [[ -e "$f" ]] || { echo "Missing required artifact: $f" >&2; exit 2; }
done
[[ ! -f "$train_dir/train.log" ]] || { echo "Output already used: $train_dir" >&2; exit 2; }
mkdir -p "$train_dir"
cp "$checkpoint" "$train_dir/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$train_dir/supermimic_deploy_config.json"
cp cfg/task/G1/G1_supermimic_surface5_direct_latent_curriculum.yaml "$train_dir/task_config.yaml"
cp dataset/passive_whole_body_full/manifest.json "$train_dir/passive_whole_body_full_manifest.json"
cp dataset/crouch_lowlevel/manifest.json "$train_dir/crouch_lowlevel_manifest.json"
cp "$0" "$train_dir/launch.sh"
cp active_adaptation/envs/mdp/supermimic_residual.py "$train_dir/action_source.py"
cp -r artifacts/surface_compliance/geometry "$train_dir/surface_geometry"
sha256sum "$checkpoint" "$deploy_config" artifacts/ceer_segmented_jacobian/geometry.json \
  dataset/passive_whole_body_full/manifest.json dataset/crouch_lowlevel/manifest.json > "$train_dir/input_sha256.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "Run root: $run_root"
  echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
  echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
  echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
  echo "Task config: $task_config"
  echo "Curriculum/action: direct tanh(action), with policy weight 0.2 -> 1.0 over progress 0.15..0.80"
  echo "Force: up to 5 motion-conditioned upper-body surface contacts; 80 N/m next-motion spring; history [0,1,2,4,8,16]"
  echo "Dataset: legacy passive_whole_body_full/crouch_lowlevel [0.8,0.2]"
  echo "Frozen SuperMimic: $checkpoint"
} | tee "$train_dir/run_info.txt"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" "task.command.surface_path=$train_dir/surface_geometry" \
  algo=ceer_force_prompt_teacher +algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" \
  "total_frames=${TOTAL_FRAMES:-4000000000}" checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=supermimic-surface5-direct-latent-curriculum \
  "wandb.run_name=supermimic-surface5-direct-latent-curriculum-legacy-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"
find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
