#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=5,6,7
export MEMPATH="$PWD/dataset" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
out="${OUTPUT_ROOT:-$PWD/outputs/$(date +%Y%m%d_%H%M%S)-supermimic-flow-latent1-compliancemix}"
[[ -f "$MEMPATH/compliance_mix_29h_v1/READY" ]]
[[ ! -e "$out/train.log" ]]
mkdir -p "$out"
"$python_bin" - "$PWD" "$out" <<'PY'
import sys,shutil,json,hashlib
from pathlib import Path
root,out=map(Path,sys.argv[1:]);code=out/'code';code.mkdir()
for name in ['active_adaptation','cfg','scripts']:
 shutil.copytree(root/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for rank in range(3):
 p=root/f'artifacts/flow_compliance/smoke/rank{rank}.json'
 report=json.loads(p.read_text())
 assert report['ddp_synchronized'] and report['low_level_frozen'] and report['checkpoint_roundtrip'] and report['flow_updated'] and report['flow_physics_steps']==32 and report['flow_horizon']==1 and report['flow_output_dim']==32
shutil.copytree(root/'artifacts/flow_compliance/smoke',out/'preflight')
shutil.copytree(root/'artifacts/surface_compliance/geometry',out/'surface_geometry')
shutil.copyfile(root/'artifacts/ceer_segmented_jacobian/geometry.json',out/'fk_geometry.json')
shutil.copyfile(root/'dataset/compliance_mix_29h_v1/manifest.json',out/'dataset_manifest.json')
assets=Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000')
(out/'lowlevel_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [assets/'ckpt_e11c_0220000.pt', assets/'deploy_config.json']},indent=2))
(code/'source_sha256.json').write_text(json.dumps({str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()},indent=2))
PY
export PYTHONPATH="$out/code:/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=3 "$out/code/scripts/train.py" \
 task=G1/G1_supermimic_surface_observable5_compliance_mix algo=supermimic_flow_latent \
 "task.command.surface_path=$out/surface_geometry" "task.command.geometry_path=$out/fk_geometry.json" \
 "task.num_envs=${NUM_ENVS_PER_GPU:-16384}" "total_frames=${TOTAL_FRAMES:-4000000000}" \
 checkpoint_path=null vecnorm=train save_interval=50 skip_final_eval=true \
 wandb.mode=online wandb.project=supermimic-flow-compliance \
 wandb.run_name=flow-latent1-compliancemix-gtforce \
 "hydra.run.dir=$out" "$@" 2>&1 | tee "$out/train.log"
