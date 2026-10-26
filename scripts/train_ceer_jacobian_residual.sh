#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/ceer-jacobian-residual/$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/train_ceer_residual.sh task=G1/G1_ceer_jacobian_compliance wandb.project=ceer-jacobian-residual-compliance "$@"
