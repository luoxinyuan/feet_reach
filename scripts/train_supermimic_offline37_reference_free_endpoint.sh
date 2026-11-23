#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$project_dir/artifacts/offline_force_baseline/train37_budget60}"
release_root="${SUPER_MIMIC_SOURCE:-/home/xl521/supermimic_force_transition}"
release_dir="$release_root/pretrained/e11c_0220000"
run_stamp="$(date +%Y%m%d_%H%M%S)"
out="${OUTPUT_ROOT:-$project_dir/outputs/${run_stamp}-supermimic-offline37-reference-free-endpoint}"

test -f "$cache/manifest.json"
test -f "$release_dir/ckpt_e11c_0220000.pt"
test -f "$release_dir/deploy_config.json"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out" >&2
  exit 2
fi

"$python_bin" "$project_dir/scripts/offline_force_baseline/validate_training_cache.py" --cache "$cache"
mkdir -p "$out"
printf '%s\n' "$out" > "$project_dir/outputs/latest_supermimic_offline37_reference_free_endpoint.txt"

cat > "$out/README.md" <<EOF
# SuperMimic reference-free residual with 37-point low-force prompts

This is a 4B-transition PPO teacher run on physical GPUs $CUDA_VISIBLE_DEVICES.
It reuses the established `train37_budget60` force cache and its 80/20 dataset
mixture, applies the cached forces physically, and gives the current 37-point
force token to the actor. The cache caps each point at 30 N, total force at
60 N, and slew at 150 N/s; measured mean summed force is about 5.39 N.

The frozen SuperMimic low level does not read future motion GT. The 58-D PPO
action contains a 29-D current-reference residual plus a 29-D future-endpoint
correction. The endpoint correction is linearly blended into a 10-frame
constant-velocity preview, with joint limits applied and qd derived from the
resulting sequence. The low-level proprioception is measured causal history.

This compares with the 3KP reference-free run while changing the force interface
and adding a learned preview endpoint. Because both changes differ, interpret
the result as a combined-method comparison, not a single-factor ablation.

Frozen low-level: released SuperMimic e11c_0220000. Environment steps:
${TOTAL_FRAMES:-4000000000}. Environments per GPU: ${NUM_ENVS_PER_GPU:-16384}.
EOF
cp "$cache/manifest.json" "$out/force_manifest.json"
cp "$cache/REPORT.md" "$out/force_cache_report.md"

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
  task=G1/G1_supermimic_offline37_reference_free_endpoint \
  algo=ceer_force_prompt_teacher \
  "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" \
  "total_frames=${TOTAL_FRAMES:-4000000000}" \
  checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
  wandb.mode=online \
  wandb.project=supermimic-reference-free-offline37 \
  wandb.run_name=supermimic-reference-free-37pt-endpoint-preview \
  "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
