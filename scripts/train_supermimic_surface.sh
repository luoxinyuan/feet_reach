#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${CONTACT_MODE:?Set CONTACT_MODE=random or optimized}"
case "$mode" in random) default_gpus=5,6,7;; optimized) default_gpus=2,3,4;; *) exit 2;; esac
# The shipped adapter consumes GT motion. Do not silently label it force-only.
if [[ "${SUPER_MIMIC_GOAL_MODE:-}" != reference_tracking ]]; then
  echo 'SuperMimic goal interface unresolved: current adapter requires reference_tracking (future GT + residual). A force-only adapter needs separate validation.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpus}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
out="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-supermimic-surface-$mode}"
if [[ -e "$out/train.log" ]]; then echo 'Output already used' >&2;exit 2;fi
mkdir -p "$out"
"$python_bin" - "$PWD" "$out" <<'PY'
import sys,shutil,hashlib,json
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);dest=out/'code';dest.mkdir(exist_ok=False)
for name in ['active_adaptation','cfg','scripts']:
 shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files=[Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt'),Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json')]
for f in files:shutil.copyfile(f,out/f.name)
shutil.copytree(root/'artifacts/surface_compliance/geometry',out/'surface_geometry')
(out/'assets_sha256.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
(dest/'source_sha256.json').write_text(json.dumps({str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()},indent=2))
PY
export PYTHONPATH="$out/code:$PYTHONPATH"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" "$out/code/scripts/train.py" \
 "task=G1/G1_supermimic_surface_$mode" algo=ceer_force_prompt_teacher \
 "task.command.surface_path=$out/surface_geometry" "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" \
 "total_frames=${TOTAL_FRAMES:-4000000000}" checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
 wandb.mode=online wandb.project=supermimic-surface-compliance \
 "wandb.run_name=surface-$mode-reference-tracking-residual" "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
