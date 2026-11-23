#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
k="${MAX_CONTACTS:?Set MAX_CONTACTS=3 or 5}"
case "$k" in 3) physical_gpus=2,3,4;; 5) physical_gpus=5,6,7;; *) exit 2;; esac
export CUDA_VISIBLE_DEVICES="$physical_gpus"
export PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="$PWD/dataset" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
out="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-surface-observable${k}-latent-teacher}"
preflight="${PREFLIGHT:?Set passing three-GPU preflight directory}"
study="${STUDY:?Set frozen force mapping study directory}"
"$python_bin" - "$preflight" "$study" <<'PY'
import json,sys
from pathlib import Path
for rank in range(3):
 r=json.loads((Path(sys.argv[1])/f'rank{rank}.json').read_text())
 assert r['ddp_synchronized'] and r['low_level_frozen'] and not r['lowlevel_uses_gt_reference']
assert json.loads(Path('artifacts/surface_contact_study/standing_fixed/standing.json').read_text())['passed']
assert (Path(sys.argv[2])/'SELECTION.json').exists()
PY
if [[ -e "$out/train.log" ]]; then echo 'Output already used' >&2;exit 2;fi
mkdir -p "$out"
"$python_bin" - "$PWD" "$out" "$preflight" "$study" <<'PY'
import sys,shutil,hashlib,json
from pathlib import Path
root,out,preflight,study=map(Path,sys.argv[1:]);dest=out/'code';dest.mkdir(exist_ok=False)
for name in ['active_adaptation','cfg','scripts']:
 shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['ckpt_e11c_0220000.pt','deploy_config.json']:
 shutil.copyfile(Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000')/name,out/name)
shutil.copytree(root/'artifacts/surface_compliance/geometry',out/'surface_geometry')
shutil.copyfile(root/'artifacts/ceer_segmented_jacobian/geometry.json',out/'fk_geometry.json')
shutil.copytree(preflight,out/'preflight');
# Full-size memory/DDP profile differs only in the .15 vs .30 coverage prior.
import os
shutil.copytree(root/f"artifacts/surface_contact_study/ddp{os.environ['MAX_CONTACTS']}_final",out/'fullsize_memory_preflight');shutil.copyfile(study/'SELECTION.json',out/'force_selection.json')
(dest/'source_sha256.json').write_text(json.dumps({str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()},indent=2))
(out/'asset_sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.suffix in ['.pt','.json']},indent=2))
PY
export PYTHONPATH="$out/code:$PYTHONPATH"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=3 "$out/code/scripts/train.py" \
 "task=G1/G1_supermimic_surface_observable$k" algo=ceer_force_prompt_teacher \
 "task.command.surface_path=$out/surface_geometry" "task.command.geometry_path=$out/fk_geometry.json" \
 "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
 checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
 wandb.mode=online wandb.project=supermimic-surface-compliance \
 "wandb.run_name=observable-${k}contact-latent-gt-teacher" "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
