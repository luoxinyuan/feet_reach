#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$project_dir/dataset"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$project_dir/outputs/wall-foot-reach/$run_stamp"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/wall-foot-reach/latest_pipeline.txt"
checkpoint_args=()
for phase in train adapt finetune; do
  run_name="${WANDB_RUN_NAME:-v1}"
  if [[ "$phase" != train ]]; then run_name="$run_name-$phase"; fi
  echo "Starting $phase: $output_dir/$phase"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train.py task=G1/G1_wall_foot_reach "+exp=$phase" \
    algo.symmetry_augmentation=false "wandb.project=${WANDB_PROJECT:-wall-foot-reach}" \
    wandb.mode=online "wandb.run_name=$run_name" "wandb.id=wall-foot-reach-${phase}-${run_stamp}" \
    "hydra.run.dir=$output_dir/$phase" "${checkpoint_args[@]}" \
    2>&1 | tee "$output_dir/$phase.log"
  mapfile -t checkpoints < <(find "$output_dir/$phase" -type f -name checkpoint_final.pt)
  if [[ ${#checkpoints[@]} -ne 1 ]]; then
    echo "Expected exactly one final checkpoint for $phase; found ${#checkpoints[@]}" >&2
    exit 1
  fi
  checkpoint_args=("checkpoint_path=${checkpoints[0]}")
  echo "Completed $phase; checkpoint: ${checkpoints[0]}"
done
echo 'All three phases completed.'
