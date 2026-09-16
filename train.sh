#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
PROJECT="your-wandb-entity/your-low-level-project"
export CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_PORT=29502
NPROC=4
SCRIPT="scripts/train.py"

# Select a low-level pipeline explicitly when launching from tmux.  The public
# default is the 30 N non-compliant 3kp backbone used by the paper.
PIPELINE="${PIPELINE:-stiff30}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

# The following low-level pipelines are the supported paper/public options:
#
#   PIPELINE=stiff30
#     Shared 3-kp stiff low-level policy with 30 N external-force probes.
#
#   PIPELINE=fixed_ee
#     End-to-end fixed 200xyz EE-compliance baseline.
#
#   PIPELINE=range_100_600
#     End-to-end independently ranged xyz compliance baseline.
#
#   PIPELINE=5kp
#     Optional 5-kp locomotion policy.
#
# Each option runs train -> adapt -> finetune.  Keep the alternatives here so
# the public repository has one low-level launcher without old one-off shell
# wrappers scattered across the root directory.
case "$PIPELINE" in
  stiff30)
    run_pipeline "G1/G1_3kp_stiff" "compliance_3kp_stiff" \
      "limmt_full_force30_${RUN_TIMESTAMP}"
    ;;
  fixed_ee)
    run_pipeline "G1/G1_3kp_ee_net_pull_force_b" "compliance_3kp_ee_fixed200" \
      "${RUN_TIMESTAMP}"
    ;;
  range_100_600)
    run_pipeline "G1/G1_3kp_ee_xyz_range_100_600" \
      "compliance_3kp_ee_xyz_range_100_600" "${RUN_TIMESTAMP}"
    ;;
  5kp)
    run_pipeline "G1/G1_5kp" "compliance_5kp" "${RUN_TIMESTAMP}"
    ;;
  *)
    echo "Unknown PIPELINE=$PIPELINE" >&2
    echo "Choose: stiff30, fixed_ee, range_100_600, or 5kp" >&2
    exit 2
    ;;
esac
