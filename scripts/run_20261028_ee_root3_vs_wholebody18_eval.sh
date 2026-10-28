#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
out="artifacts/20261028-ee-root3-vs-wholebody18-compliance"
dataset="../artifacts/ceer_single_hand_ee/eval_cache_unseen_20261025/dataset"
three_cache="artifacts/20261028-ee-root3-unseen-eval"
whole_cache="artifacts/ceer_single_hand_ee/eval_cache_unseen_20261025"
three_ckpt="outputs/ceer-segmented-jacobian-residual/20261027_173438_ee-root3/wandb/run-20261027_173540-a88dyzbq/files/checkpoint_final.pt"
whole_ckpt="outputs/ceer-segmented-jacobian-residual/multi10_20261026_163438/wandb/run-20261026_163534-4vqahv90/files/checkpoint_final.pt"

for variant in wholebody18 ee_root3; do
  if [[ "$variant" == wholebody18 ]]; then ckpt="$whole_ckpt"; cache="$whole_cache"; else ckpt="$three_ckpt"; cache="$three_cache"; fi
  for scenario in guided zero; do
    "$python_bin" scripts/eval_passive_policies.py --task whole-body --controller student \
      --scenario "$scenario" --num-envs 256 --steps 500 --seed 20261031 \
      --checkpoint "$ckpt" --dataset-path "$dataset" --guidance-cache "$cache" \
      --output "$out/$variant/$scenario"
  done
done
