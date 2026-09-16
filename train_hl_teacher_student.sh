#!/usr/bin/env bash
set -euo pipefail

# Public paper training entry point. The shared low-level policy is frozen
# while each high-level teacher/student policy is trained independently.
LOW_PROJECT_PATH="${LOW_PROJECT_PATH:-your-wandb-entity/your-low-level-project}"
HL_PROJECT_PATH="${HL_PROJECT_PATH:-your-wandb-entity/your-high-level-project}"
HL_WANDB_PROJECT="${HL_WANDB_PROJECT:-your-high-level-project}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29502}"
NPROC="${NPROC:-4}"
LOW_RUN_PATH="${LOW_RUN_PATH:-${LOW_PROJECT_PATH}/compliance_3kp_stiff_finetune_force30}"
LOW_CHECKPOINT_PATH="${LOW_CHECKPOINT_PATH:-}"
TEACHER_FRAMES="${TEACHER_FRAMES:-4000_000_000}"
ADAPT_FRAMES="${ADAPT_FRAMES:-1000_000_000}"
TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ONLY="${RUN_ONLY:-${1:-all}}"

case "$RUN_ONLY" in
  all|experts|baselines) ;;
  *)
    echo "Usage: $0 [all|experts|baselines]" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES

run_stage() {
  local task="$1"
  local algo="$2"
  local run_id="$3"
  local total_frames="$4"
  local checkpoint_path="${5:-}"

  local cmd=(torchrun
    --nproc_per_node="$NPROC"
    --master_port="$MASTER_PORT"
    scripts/train.py
    task="$task"
    algo="$algo"
    total_frames="$total_frames"
    wandb.project="$HL_WANDB_PROJECT"
    wandb.id="$run_id"
  )

  if [[ -n "$LOW_CHECKPOINT_PATH" ]]; then
    cmd+=(
      task.action.low_policy.checkpoint_path="$LOW_CHECKPOINT_PATH"
      task.action.low_policy.run_path=null
    )
  else
    cmd+=(task.action.low_policy.run_path="$LOW_RUN_PATH")
  fi

  if [[ -n "$checkpoint_path" ]]; then
    cmd+=(checkpoint_path="$checkpoint_path")
  fi

  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

run_pipeline() {
  local group="$1"
  local task="$2"
  local run_name="$3"
  local teacher_run_id="${run_name}_teacher_${TIMESTAMP}"
  local adapt_run_id="${run_name}_adapt_${TIMESTAMP}"

  echo "=== ${group}: ${task} ==="
  run_stage "$task" root_student_force_ppo "$teacher_run_id" "$TEACHER_FRAMES"
  run_stage "$task" root_student_force_ppo_adapt "$adapt_run_id" "$ADAPT_FRAMES" \
    "run:${HL_PROJECT_PATH}/${teacher_run_id}"
}

run_if_selected() {
  local group="$1"
  shift
  if [[ "$RUN_ONLY" == "all" || "$RUN_ONLY" == "$group" ]]; then
    run_pipeline "$group" "$@"
  fi
}

# Ten experts used by the final 100--600 N/m analytical MoE.
run_if_selected experts G1/hl/ee/G1_hl_ee_x100_compliance_pos_delta_force_b_student \
  ee_x100_3kp_force30_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_y100_compliance_pos_delta_force_b_student \
  ee_y100_3kp_force30_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_z100_direct_compliance_pos_delta_force_b_student \
  ee_z100_direct_3kp_force30_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_x_compliance_pos_delta_force_b_student \
  ee_x200_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_y_compliance_pos_delta_force_b_student \
  ee_y200_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_z_compliance_pos_delta_force_b_student \
  ee_z200_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_x400_compliance_pos_delta_force_b_student \
  ee_x400_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_y400_compliance_pos_delta_force_b_student \
  ee_y400_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_z400_compliance_pos_delta_force_b_student \
  ee_z400_3kp_force_b_stu
run_if_selected experts G1/hl/ee/G1_hl_ee_xyz600_posscale035_direct_compliance_pos_delta_force_b_student \
  ee_xyz600_posscale035_direct_3kp_force30_stu

# Matched fixed and ranged high-level baselines.
run_if_selected baselines G1/hl/ee/G1_hl_ee_compliance_pos_delta_force_b_student \
  ee_fixed_200xyz_3kp_force_b_stu
run_if_selected baselines G1/hl/ee/G1_hl_ee_xyz_range_100_600_force_b_student \
  ee_xyz_range_100_600_3kp_force_b_stu

# Optional root high-level student policies. Uncomment a line when a root
# compliance policy is needed; these are intentionally not part of the
# default EE expert/baseline run.
# run_pipeline root G1/hl/root/G1_hl_force_walk_B60_force_b_student \
#   root_force_walk_B60
# run_pipeline root G1/hl/root/G1_hl_force_walk_B200_force_b_student \
#   root_force_walk_B200
# run_pipeline root G1/hl/root/G1_hl_root_hold_force_b_student \
#   root_hold
