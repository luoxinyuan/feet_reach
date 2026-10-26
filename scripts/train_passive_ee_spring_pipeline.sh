#!/usr/bin/env bash
set -euo pipefail
export EXPERIMENT_NAME=passive-ee-spring
export TASK=G1/G1_passive_ee_spring
export TEACHER_GUIDANCE=true
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_passive_ee_dense_pipeline.sh"
