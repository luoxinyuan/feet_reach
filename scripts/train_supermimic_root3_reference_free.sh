#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin="${PYTHON_BIN:-/home/xl521/software/miniconda3/envs/gentle/bin/python}"
out="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-supermimic-root3-reference-free}"
if [[ -e "$out/train.log" ]]; then echo 'Output already used' >&2;exit 2;fi
"$python_bin" - "$PWD" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);base=root/'artifacts/reference_free_supermimic'
s=json.loads((base/'standing/standing.json').read_text());assert s['passed'] and s['lowlevel_dataset_access_blocked']
for rank in range(3):
 r=json.loads((base/f'fullscale/rank{rank}.json').read_text())
 assert r['lowlevel_dataset_access_blocked'] and r['lowlevel_action_identical_with_dataset_blocked'] and r['ddp_synchronized'] and r['low_level_frozen']
print('Reference-free standing, information-isolation and three-GPU PPO gates passed')
PY
mkdir -p "$out"
"$python_bin" - "$PWD" "$out" <<'PY'
import sys,shutil,hashlib,json
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);dest=out/'code';dest.mkdir(exist_ok=False)
for name in ['active_adaptation','cfg','scripts']:
 shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files=[Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt'),Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json')]
for f in files:shutil.copyfile(f,out/f.name)
shutil.copytree(root/'artifacts/reference_free_supermimic',out/'preflight')
(out/'assets_sha256.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
(dest/'source_sha256.json').write_text(json.dumps({str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()},indent=2))
PY
export PYTHONPATH="$out/code:$PYTHONPATH"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" "$out/code/scripts/train.py" \
 task=G1/G1_supermimic_root3_reference_free algo=ceer_dense_force algo.phase=teacher \
 "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
 checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
 wandb.mode=online wandb.project=supermimic-online-root3-crouch20-force-history \
 wandb.run_name=supermimic-root3-reference-free-force-history \
 "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
