#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export PYTHONPATH="${ISAACLAB_ROOT:-/home/xl521/IsaacLab}/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
cache="${FORCE_CACHE:-$PWD/artifacts/offline_force_baseline/train37_budget60}"
output="${OUTPUT_ROOT:-$PWD/outputs/20261105-ceer-offline-force-prompt-teacher}"
test -f "$cache/manifest.json"
"$python_bin" scripts/offline_force_baseline/validate_training_cache.py --cache "$cache"
if [[ -f "$output/train.log" ]]; then echo "Output already used: $output" >&2; exit 2; fi
mkdir -p "$output"
cp "$cache/manifest.json" "$output/force_manifest.json"
lowlevel="$PWD/outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt"
sha256sum "$lowlevel" > "$output/frozen_lowlevel.sha256"
# Isolate this run from subsequent edits in the shared working tree.
"$python_bin" - "$PWD" "$output/code" <<'PYCODE'
import sys,shutil,hashlib,json
from pathlib import Path
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=False)
for name in ['active_adaptation','scripts','cfg']:
    shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
hashes={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.sh']}
(dest/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
PYCODE
export PYTHONPATH="$output/code:$PYTHONPATH"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" "$output/code/scripts/train.py" \
 "task=${TASK_NAME:-G1/G1_ceer_offline_force_prompt_teacher}" "algo=${ALGO_NAME:-ceer_force_prompt_teacher}" \
 "task.command.guidance_cache=$cache" "task.num_envs=${NUM_ENVS_PER_GPU:-20480}" \
 "total_frames=${TOTAL_FRAMES:-4000000000}" checkpoint_path=null vecnorm=train save_interval=50 \
 "wandb.mode=${WANDB_MODE:-online}" wandb.project=ceer-force-prompt-teacher \
 "wandb.run_name=${WANDB_RUN_NAME:-offline37-budget60-no-physical-force-no-force-history}" \
 "hydra.run.dir=$output" "$@" 2>&1 | tee "$output/train.log"
