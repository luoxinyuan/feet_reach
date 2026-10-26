#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
case "${1:?Expected ceer or spring-direct}" in
  ceer) name=passive-ee-ceer; task=G1/G1_passive_ee_ceer; algo=ppo_train; extra=(algo.symmetry_augmentation=false); export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}";;
  spring-direct) name=passive-ee-spring-direct; task=G1/G1_passive_ee_spring; algo=passive_spring_direct; extra=(); export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}";;
  *) exit 2;;
esac
output_dir="$PWD/outputs/$name/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$PWD/outputs/$name/latest_pipeline.txt"
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
 scripts/train.py "task=$task" "algo=$algo" "${extra[@]}" \
 "task.num_envs=${NUM_ENVS:-12288}" "total_frames=${TEACHER_FRAMES:-1200000000}" \
 save_interval=50 "wandb.mode=${WANDB_MODE:-online}" "wandb.project=$name" \
 "hydra.run.dir=$output_dir/train" 2>&1 | tee "$output_dir/train.log"
for scenario in guided hold; do
 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$python_bin" scripts/eval_passive_policies.py \
 --task compliance --run-dir "$output_dir" --controller teacher --scenario "$scenario" \
 --output "$output_dir/teacher_validation/$scenario" > "$output_dir/eval_$scenario.log" 2>&1
done
"$python_bin" scripts/check_passive_teacher_quality.py "$output_dir/teacher_validation" 2>&1 | tee "$output_dir/teacher_validation.log"
