#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
output_dir="${OUTPUT_DIR:-$PWD/outputs/sonic-residual/$(date +%Y%m%d_%H%M%S)}"
if [[ ! -f .cache/sonic/manifest.json ]]; then
  echo "Missing SONIC resources. Run: $python_bin scripts/sonic/prepare.py" >&2
  exit 1
fi
mkdir -p "$output_dir"
cp .cache/sonic/manifest.json "$output_dir/sonic_manifest.json"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpu_list[@]}" \
  scripts/train.py task=G1/G1_sonic_compliance algo=sonic_residual \
  "task.num_envs=${NUM_ENVS:-2048}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  save_interval=50 "wandb.mode=${WANDB_MODE:-online}" wandb.project=sonic-residual-compliance \
  "wandb.run_name=sonic-residual-$(basename "$output_dir")" \
  "hydra.run.dir=$output_dir" "$@" 2>&1 | tee "$output_dir/train.log"
