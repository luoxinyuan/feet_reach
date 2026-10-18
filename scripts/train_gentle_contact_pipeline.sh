#!/usr/bin/env bash
# Matched gentle-contact A/B experiments, teacher PPO then adaptation.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
variant="${GENTLE_VARIANT:-A}"
case "$variant" in
 A) default_gpus=0,1,2,3 ;;
 B) default_gpus=4,5,6,7 ;;
 *) echo 'GENTLE_VARIANT must be A or B' >&2; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpus}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$project_dir/dataset"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
low_checkpoint="${LOW_CHECKPOINT:-$project_dir/.cache/ceer/checkpoint_final.pt}"
if [[ ! -f "$low_checkpoint" ]]; then
  echo "Missing CEER checkpoint: $low_checkpoint" >&2
  exit 1
fi
low_checkpoint="$(realpath "$low_checkpoint")"
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$project_dir/outputs/gentle-contact/$variant/$run_stamp"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$project_dir/outputs/gentle-contact/$variant/latest_pipeline.txt"
sha256sum "$low_checkpoint" > "$output_dir/low_checkpoint.sha256"
cp "cfg/task/G1/G1_gentle_$variant.yaml" "$output_dir/task_variant.yaml"
printf '%s\n' "variant=$variant GPUs=$CUDA_VISIBLE_DEVICES envs_per_gpu=${NUM_ENVS:-4096}" > "$output_dir/experiment.txt"
mkdir -p "$output_dir/source"
cp active_adaptation/envs/mdp/commands/gentle_residual.py active_adaptation/utils/gentle_contact.py "$output_dir/source/"
checkpoint_args=()
for phase in train adapt; do
  frames="${TEACHER_FRAMES:-4000000000}"
  norm=train
  curriculum=true
  if [[ "$phase" == adapt ]]; then frames="${ADAPT_FRAMES:-1000000000}"; norm=eval; curriculum=false; fi
  echo "Starting $phase: $output_dir/$phase"
  "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC:-4}" \
    scripts/train.py "task=G1/G1_gentle_$variant" algo=proactive_residual \
    "algo.phase=$phase" "task.command.gentle.curriculum_enabled=$curriculum" "total_frames=$frames" "vecnorm=$norm" \
    "task.num_envs=${NUM_ENVS:-4096}" \
    "task.action.low_policy.checkpoint_path=$low_checkpoint" \
    "wandb.project=${WANDB_PROJECT:-wall-foot-reach}" wandb.mode=online \
    "wandb.run_name=${WANDB_RUN_NAME:-gentle-contact-$variant-v1}-$phase" \
    "wandb.id=gentle-${variant}-${phase}-${run_stamp}" \
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
echo 'Teacher and adaptation completed; CEER remained frozen throughout.'
