#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXPERIMENT_NAME=passive-ee-dense-guided
export TEACHER_GUIDANCE=true
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_passive_ee_dense_pipeline.sh"
