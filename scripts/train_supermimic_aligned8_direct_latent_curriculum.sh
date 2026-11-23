#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

encoder="${FORCE_ENCODER_PATH:-}"
task=G1/G1_supermimic_aligned8_direct_latent_curriculum
project=supermimic-aligned8-direct-latent-curriculum
default_gpus=6,7
if [[ -n "$encoder" ]]; then
  task=G1/G1_supermimic_aligned8_force_latent_curriculum
  project=supermimic-aligned8-force-latent-curriculum
  default_gpus=4,5
fi
project="${WANDB_PROJECT:-$project}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpus}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-$project-gpu${CUDA_VISIBLE_DEVICES//,/}}"
cache="${FORCE_CACHE:-$PWD/artifacts/surface_contact_study/coverage000_boundary_fixed_20261122/aligned_training_cache_legacy}"
base="${SUPER_MIMIC_ROOT:-/home/xl521/supermimic_force_transition}/pretrained/e11c_0220000"
checkpoint="$base/ckpt_e11c_0220000.pt"
deploy_config="$base/deploy_config.json"
train="$root/train"
for file in "$cache/manifest.json" "$checkpoint" "$deploy_config"; do
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 2; }
done
[[ ! -f "$train/train.log" ]] || { echo "Output already used: $train" >&2; exit 2; }
mkdir -p "$train"
cp "cfg/task/$task.yaml" "$train/task_config.yaml"
cp "$cache/manifest.json" "$train/force_cache_manifest.json"
cp "$0" "$train/launch.sh"
sha256sum "$checkpoint" "$deploy_config" "$cache/manifest.json" > "$train/input_sha256.txt"
encoder_args=()
if [[ -n "$encoder" ]]; then
  [[ -f "$encoder" ]] || { echo "Missing force encoder $encoder" >&2; exit 2; }
  cp "$encoder" "$train/force_encoder.pt"
  sha256sum "$train/force_encoder.pt" >> "$train/input_sha256.txt"
  encoder_args=("task.command.force_encoder_path=$train/force_encoder.pt")
fi
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
printf '%s\n' "GPUs=$CUDA_VISIBLE_DEVICES" "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU:-16384}" \
  "TOTAL_FRAMES=${TOTAL_FRAMES:-4000000000}" "FORCE_CACHE=$cache" > "$train/run_info.txt"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task" "task.command.force_cache=$cache" "${encoder_args[@]}" \
  algo=ceer_force_prompt_teacher +algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" "wandb.project=$project" \
  "wandb.run_name=$(basename "$root")" "hydra.run.dir=$train" "$@" 2>&1 | tee "$train/train.log"
find "$train" -type f -name checkpoint_final.pt -print -quit > "$root/final_checkpoint.txt"
