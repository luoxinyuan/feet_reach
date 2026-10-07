#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="${ISAACLAB_PATH:-/home/xl521/IsaacLab}/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x /home/xl521/software/miniconda3/envs/gentle/bin/python ]]; then
    python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
  else
    python_bin=python
  fi
fi
exec "$python_bin" -u scripts/eval_foot_reach.py "$@"
