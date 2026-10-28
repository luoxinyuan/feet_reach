#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-online-ee-root3-residual/$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash scripts/train_ceer_residual.sh \
  task=G1/G1_ceer_online_ee_root3_compliance \
  wandb.project=ceer-online-ee-root3-residual-compliance \
  "wandb.run_name=online-ee-root3-$(basename "$OUTPUT_DIR")" "$@"
