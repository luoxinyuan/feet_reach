#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_sonic_online_estimated_9pt_crouch20_force_history}"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d)-sonic-online-estimated-9pt-crouch20-force-history/$(date +%H%M%S)}"
train_dir="$run_root/train"

if [[ ! -f .cache/sonic/manifest.json || ! -f .cache/sonic/actor.pt ]]; then
  echo "Missing validated SONIC resources. Run scripts/sonic/prepare.py first." >&2
  exit 2
fi
if [[ -f "$train_dir/train.log" ]]; then
  echo "Output already used: $train_dir" >&2
  exit 2
fi

mkdir -p "$train_dir"
cp .cache/sonic/manifest.json "$train_dir/sonic_manifest.json"
sha256sum .cache/sonic/actor.pt > "$train_dir/frozen_sonic_actor.sha256"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
echo "Task config: $task_config"
echo "Force points: 9 upper-body points"
echo "Force history offsets: [0, 1, 2, 4, 8, 16]"
echo "Dataset mixture: 0.8 whole-body motion, 0.2 crouch"
echo "Frozen SONIC actor: .cache/sonic/actor.pt"
echo "Upper actor output: 29-DOF joint-reference residual"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" \
  algo=ceer_dense_force algo.phase=teacher \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=sonic-online-estimated-9pt-crouch20-force-history \
  "wandb.run_name=sonic-online-estimated-9pt-crouch20-force-history-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"

find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
