#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
release_root="${SUPER_MIMIC_SOURCE:-/home/xl521/supermimic_force_transition}"
release_dir="$release_root/pretrained/e11c_0220000"
run_stamp="$(date +%Y%m%d_%H%M%S)"
out="${OUTPUT_ROOT:-$project_dir/outputs/${run_stamp}-supermimic-root3-reference-free-endpoint-interpolation}"

test -f "$release_dir/ckpt_e11c_0220000.pt"
test -f "$release_dir/deploy_config.json"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out" >&2
  exit 2
fi

mkdir -p "$out"
printf '%s\n' "$out" > "$project_dir/outputs/latest_supermimic_root3_endpoint_interpolation.txt"
cat > "$out/README.md" <<EOF
# SuperMimic reference-free endpoint-interpolation policy

This is a 4B-transition PPO teacher run on physical GPUs $CUDA_VISIBLE_DEVICES.
It retains the existing online 3KP (root + both hands) force generator,
force-history tokens, motion mixture, rewards, and teacher force input. The
frozen low level is released SuperMimic e11c_0220000.

The PPO action is 29-D. At every step it directly predicts a joint endpoint
offset from measured current joint position, bounded to 0.36 rad per joint.
SuperMimic receives ten future goal frames formed by linearly interpolating
from the measured current pose to that endpoint. There is no persistent target
integration, constant-velocity extrapolation, future motion GT, or per-frame
policy prediction. This is the 3KP endpoint-only comparison to the current
37-point endpoint-preview run.

Environment steps: ${TOTAL_FRAMES:-4000000000}. Environments per GPU:
${NUM_ENVS_PER_GPU:-16384}. Checkpoint interval: 50 updates.
EOF

frozen_root="$out/frozen_supermimic"
mkdir -p "$frozen_root/pretrained/e11c_0220000"
cp "$release_dir/ckpt_e11c_0220000.pt" "$frozen_root/pretrained/e11c_0220000/"
cp "$release_dir/deploy_config.json" "$frozen_root/pretrained/e11c_0220000/"
cp -a "$release_root/code" "$frozen_root/code"
export SUPER_MIMIC_ROOT="$frozen_root"
sha256sum "$release_dir/ckpt_e11c_0220000.pt" "$release_dir/deploy_config.json" > "$out/supermimic_sha256.txt"

"$python_bin" - "$project_dir" "$out/code" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root, dest = Path(sys.argv[1]), Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=False)
for name in ('active_adaptation', 'cfg', 'scripts'):
    shutil.copytree(root / name, dest / name,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in dest.rglob('*') if p.is_file() and not p.is_symlink()
          and p.suffix in {'.py', '.yaml', '.sh'}}
(dest / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
PY

export PYTHONPATH="$out/code:/home/xl521/IsaacLab/source/isaaclab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpu_list[@]}" \
  "$out/code/scripts/train.py" \
  task=G1/G1_supermimic_root3_reference_free_endpoint_interpolation \
  algo=ceer_dense_force algo.phase=teacher \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" \
  "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  wandb.mode=online \
  wandb.project=supermimic-reference-free-root3 \
  wandb.run_name=supermimic-root3-endpoint-only-interpolation-v2 \
  "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
