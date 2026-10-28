#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
points_cfg="cfg/task/G1/G1_ceer_ee_root3_compliance.yaml"
source_ds="artifacts/ceer_segmented_jacobian/train18_single_hand_ee_static_screened_multi10_20261026/dataset"
cache_root="artifacts/ceer_segmented_jacobian"
base_cache="$cache_root/20261027-ee-root3-base"
multi_cache="$cache_root/20261027-ee-root3-multi10"

if [[ ! -f "$base_cache/manifest.json" ]]; then
  "$python_bin" scripts/data_process/prepare_segmented_guidance.py \
    --source-dataset "$source_ds" --points-config "$points_cfg" --points 3 \
    --single-hand-source artifacts/ceer_single_hand_ee/full_20261025 \
    --workers "${CACHE_WORKERS:-24}" --out "$base_cache"
fi

if [[ ! -f "$multi_cache/variant_manifest.json" ]]; then
  "$python_bin" scripts/data_process/generate_segmented_force_variants.py \
    --source-cache "$base_cache" --points-config "$points_cfg" \
    --workers "${VARIANT_WORKERS:-24}" --output "$multi_cache"
  "$python_bin" scripts/data_process/supplement_segmented_force_variants.py \
    --cache "$multi_cache" --source-cache "$base_cache" --points-config "$points_cfg" \
    --workers "${VARIANT_WORKERS:-24}"
fi

"$python_bin" scripts/references/validate_segmented_cache.py "$multi_cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-segmented-jacobian-residual/$(date +%Y%m%d_%H%M%S)_ee-root3}"
export NUM_ENVS="${NUM_ENVS:-16384}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash scripts/train_ceer_segmented_ee_root3_residual.sh
