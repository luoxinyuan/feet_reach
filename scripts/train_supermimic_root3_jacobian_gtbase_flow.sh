#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$PWD/dataset}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
base_run="$PWD/outputs/20261115-supermimic-root3-crouch20-force-history-latent/095800/train"
checkpoint="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
deploy_config="/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json"
preflight="$base_run/preflight.json"
run_root="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-supermimic-3kp-jacobian-gtbase-flow-567}"
train_dir="$run_root/train"

for f in "$checkpoint" "$deploy_config" "$preflight" "$base_run/.hydra/config.yaml"; do
  [[ -f "$f" ]] || { echo "Missing required baseline artifact: $f" >&2; exit 2; }
done
[[ ! -e "$train_dir/train.log" ]] || { echo "Output already used: $train_dir" >&2; exit 2; }

"$python_bin" - "$preflight" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
if s["fall_clips"] != 0 or s["clip_mean_joint_mae_rad"] > 0.25:
    raise SystemExit(f"SuperMimic preflight failed: {s}")
print("SuperMimic 3KP preflight passed:", s)
PY

mkdir -p "$train_dir"
cp "$checkpoint" "$train_dir/supermimic_frozen_checkpoint.pt"
cp "$deploy_config" "$train_dir/supermimic_deploy_config.json"
cp "$preflight" "$train_dir/preflight.json"
cp "$base_run/.hydra/config.yaml" "$train_dir/baseline_ppo_resolved_config.yaml"
sha256sum "$checkpoint" "$deploy_config" > "$train_dir/supermimic_sha256.txt"

# Freeze the exact code/config used by this run for reproducibility.
for name in active_adaptation cfg scripts; do
  mkdir -p "$run_root/code"
  cp -a "$name" "$run_root/code/"
done

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
echo "Run root: $run_root"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}"
echo "Total transitions: ${TOTAL_FRAMES:-4000000000}"
echo "Baseline: GT 10-frame motion latent + 0.1-scaled latent residual"
echo "Force interface: online 3KP Jacobian, history [0,1,2,4,8,16]"
echo "Change: DenseForcePPO -> FlowLatentPPO only; same dataset/rewards/action adapter"

export PYTHONPATH="$run_root/code:/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" "$run_root/code/scripts/train.py" \
  task=G1/G1_supermimic_online_estimated_root3_crouch20_force_history_latent \
  algo=supermimic_flow_latent +algo.symmetry_augmentation=false \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  wandb.mode=online wandb.project=supermimic-online-root3-crouch20-force-history \
  "wandb.run_name=supermimic-3kp-jacobian-gtbase-flow-$(basename "$run_root")" \
  "hydra.run.dir=$train_dir" "$@" 2>&1 | tee "$train_dir/train.log"

find "$train_dir" -type f -name 'checkpoint_final.pt' -print -quit > "$run_root/final_checkpoint.txt"
echo "Training finished: $run_root"
