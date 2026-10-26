#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
output_dir="$PWD/outputs/passive-whole-body/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > outputs/passive-whole-body/latest_pipeline.txt
checkpoint=()
for phase in train adapt finetune; do
 case "$phase" in
  train) frames="${TEACHER_FRAMES:-4000000000}"; algo=ppo_train; controller=teacher;;
  adapt) frames="${ADAPT_FRAMES:-100000000}"; algo=ppo_adapt; controller=adapt;;
  finetune) frames="${STUDENT_FRAMES:-4000000000}"; algo=ppo_finetune; controller=student;;
 esac
 "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train.py task=G1/G1_passive_whole_body "algo=$algo" algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS:-12288}" "total_frames=$frames" save_interval=50 \
  "wandb.mode=${WANDB_MODE:-online}" wandb.project=passive-whole-body \
  "wandb.run_name=whole-body-$phase-$(basename "$output_dir")" \
  "hydra.run.dir=$output_dir/$phase" "${checkpoint[@]}" 2>&1 | tee "$output_dir/$phase.log"
 mapfile -t checkpoints < <(find "$output_dir/$phase" -name checkpoint_final.pt -type f)
 [[ ${#checkpoints[@]} -eq 1 ]] || exit 1
 checkpoint=("checkpoint_path=${checkpoints[0]}")
 for scenario in guided zero; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$python_bin" scripts/eval_passive_policies.py \
   --task whole-body --run-dir "$output_dir" --controller "$controller" --scenario "$scenario" \
   --steps 1000 --output "$output_dir/${phase}_validation/$scenario" \
   > "$output_dir/eval_${phase}_${scenario}.log" 2>&1
 done
 if [[ "$phase" == train ]]; then
  "$python_bin" scripts/check_whole_body_teacher.py "$output_dir/train_validation" \
   2>&1 | tee "$output_dir/teacher_gate.log"
 fi
done
