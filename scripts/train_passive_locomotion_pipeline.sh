#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$project_dir/dataset"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$project_dir/outputs/passive-locomotion/$run_stamp"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/passive-locomotion/latest_pipeline.txt"
checkpoint_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then checkpoint_args=("checkpoint_path=$RESUME_CHECKPOINT"); fi
for phase in train adapt finetune; do
  case "$phase" in
    train) frames="${TEACHER_FRAMES:-400000000}";;
    adapt) frames="${ADAPT_FRAMES:-100000000}";;
    finetune) frames="${STUDENT_FRAMES:-400000000}";;
  esac
  phase_frames="$frames"
  done_frames=0
  if [[ "$phase" == train ]]; then done_frames="${TEACHER_DONE_FRAMES:-0}"; fi
  frames=$((phase_frames - done_frames))
  if ((frames <= 0)); then echo "Invalid remaining frame budget" >&2; exit 1; fi
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train.py task=G1/G1_passive_locomotion algo=passive_compliance \
    "algo.phase=$phase" "task.num_envs=${NUM_ENVS:-12288}" "total_frames=$frames" \
    "+resume_frames=$done_frames" "+phase_frames=$phase_frames" "save_interval=${SAVE_INTERVAL:-50}" \
    "wandb.project=${WANDB_PROJECT:-passive-locomotion}" "wandb.mode=${WANDB_MODE:-online}" \
    "wandb.run_name=passive-locomotion-$phase-$run_stamp" "wandb.id=passive-$phase-$run_stamp" \
    "hydra.run.dir=$output_dir/$phase" "${checkpoint_args[@]}" \
    2>&1 | tee "$output_dir/$phase.log"
  mapfile -t checkpoints < <(find "$output_dir/$phase" -type f -name checkpoint_final.pt)
  [[ ${#checkpoints[@]} -eq 1 ]] || { echo "Missing unique checkpoint: $phase" >&2; exit 1; }
  checkpoint_args=("checkpoint_path=${checkpoints[0]}")
done
