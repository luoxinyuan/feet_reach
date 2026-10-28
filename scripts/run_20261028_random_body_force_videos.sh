#!/usr/bin/env bash
set -euo pipefail
cd /home/xl521/feet_reach
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH=/home/xl521/feet_reach/dataset
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
"$python_bin" scripts/eval_random_body_compliance.py \
  --checkpoint outputs/ceer-segmented-jacobian-residual/20261027_173438_ee-root3/wandb/run-20261027_173540-a88dyzbq/files/checkpoint_final.pt \
  --output artifacts/20261028-random-body-force-compliance/ee_root3 --seed 20261028
"$python_bin" scripts/eval_random_body_compliance.py \
  --checkpoint outputs/ceer-segmented-jacobian-residual/multi10_20261026_163438/wandb/run-20261026_163534-4vqahv90/files/checkpoint_final.pt \
  --output artifacts/20261028-random-body-force-compliance/wholebody18 --seed 20261028
