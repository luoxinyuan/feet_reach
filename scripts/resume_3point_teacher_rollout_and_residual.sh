#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
run_root="$PWD/outputs/20261104-ceer-lowlevel-force-root3-teacher-supervised-residual/full"
teacher_checkpoint="$(cat "$run_root/teacher_checkpoint.txt")"
rollout_dir="$run_root/teacher_rollout"
residual_dir="$run_root/residual_teacher"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

[[ -f "$teacher_checkpoint" ]] || { echo "Missing teacher checkpoint: $teacher_checkpoint" >&2; exit 2; }
mkdir -p "$rollout_dir" "$residual_dir"

echo "Rollout: 4B transitions; ${ROLLOUT_ENVS:-16384} environments per GPU; GPUs $CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/collect_3point_lowlevel_deltaq.py \
  --checkpoint "$teacher_checkpoint" --output-dir "$rollout_dir" \
  --total-transitions 4000000000 --num-envs "${ROLLOUT_ENVS:-16384}" \
  --seed 20261104 --expected-crouch-weight 0.05 \
  2>&1 | tee "$rollout_dir/collect.log"

echo "Supervised residual teacher training"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train_force_to_deltaq_residual_streaming.py \
  --data-dir "$rollout_dir" --output-dir "$residual_dir" --batch-size 16384 --seed 20261104 \
  2>&1 | tee "$residual_dir/train.log"

printf 'Pipeline complete.\nTeacher checkpoint: %s\nTeacher rollout data: %s\nResidual teacher: %s\n' \
  "$teacher_checkpoint" "$rollout_dir" "$residual_dir/residual_teacher.pt" > "$run_root/PIPELINE_COMPLETE.txt"
echo "PIPELINE_COMPLETE $run_root"
