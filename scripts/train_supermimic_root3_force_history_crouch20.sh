#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
task_config="${TASK_CONFIG:-G1/G1_supermimic_online_estimated_root3_crouch20_force_history}"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
run_root="${OUTPUT_ROOT:-$PWD/outputs/20261109-supermimic-root3-crouch20-force-history/$(date +%H%M%S)}"
train_dir="$run_root/train"
preflight="${PREFLIGHT_JSON:-$PWD/artifacts/compliance_eval/20261109-supermimic-root3-crouch20-force-history/preflight-crouch-final/preflight.json}"

if [[ ! -f "$checkpoint" || ! -f "$deploy_config" ]]; then
  echo "Missing released SuperMimic checkpoint/deploy config." >&2
  exit 2
fi
if [[ ! -f "$preflight" ]]; then
  echo "Missing passing SuperMimic Jacobian preflight report: $preflight" >&2
  exit 2
fi
"$python_bin" - "$preflight" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r["summary"]
if s["fall_clips"] != 0 or s["clip_mean_joint_mae_rad"] > 0.25:
    raise SystemExit(f"Preflight failed stability/tracking gate: {s}")
print("Preflight gate passed:", s)
PY
if [[ -f "$train_dir/train.log" ]]; then
  echo "Output already used: $train_dir" >&2
  exit 2
fi

mkdir -p "$train_dir"
cp "$checkpoint" "$train_dir/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$train_dir/supermimic_deploy_config.json"
cp "$preflight" "$train_dir/preflight.json"
sha256sum "$checkpoint" "$deploy_config" > "$train_dir/supermimic_sha256.txt"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
echo "Task config: $task_config"
echo "Frozen SuperMimic: $checkpoint"
echo "Force interface: 3KP online Jacobian (root + left/right hand)"
echo "Dataset weights: [0.8, 0.2] (Multi10, crouch)"
echo "Force history offsets: [0, 1, 2, 4, 8, 16]"
echo "Training: teacher PPO only; no force estimator/adapt/student stage"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" scripts/train.py \
  "task=$task_config" \
  algo=ceer_dense_force algo.phase=teacher \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  "wandb.mode=${WANDB_MODE:-online}" \
  wandb.project=supermimic-online-root3-crouch20-force-history \
  "wandb.run_name=supermimic-root3-crouch20-force-history-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"

find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
