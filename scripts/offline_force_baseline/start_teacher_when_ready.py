"""Launch the authorized teacher after complete cache and 3-rank capacity audit."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
cache=ROOT/'artifacts/offline_force_baseline/train37_budget60'
bench=ROOT/'artifacts/offline_force_baseline/prompt_bench_20480'
out=ROOT/'outputs/20261105-ceer-offline-force-prompt-teacher'
status=ROOT/'artifacts/offline_force_baseline/prompt_training_status.json'
status.write_text(json.dumps(dict(state='waiting_for_cache_and_capacity_audit',cache=str(cache),bench=str(bench))))
while True:
 try:
  ready=json.loads((cache/'manifest.json').read_text()).get('complete',False)
 except (FileNotFoundError,json.JSONDecodeError):ready=False
 if ready and all((bench/f'rank{i}.json').exists() for i in range(3)):break
 time.sleep(20)
m=json.loads((cache/'manifest.json').read_text());assert m['complete']
for i in range(3):
 r=json.loads((bench/f'rank{i}.json').read_text());assert r['num_envs']==20480 and r['ddp_synchronized'] and not r['physical_force'] and not r['force_history'] and r['low_level_frozen']
out.mkdir(parents=True,exist_ok=True)
if (out/'train.log').exists():raise RuntimeError('Output already used; refusing duplicate launch')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3,4',NUM_ENVS_PER_GPU='20480',OUTPUT_ROOT=str(out),WANDB_MODE='online')
with (out/'launcher.log').open('w') as log:
 p=subprocess.Popen(['bash','scripts/train_ceer_offline_force_prompt_teacher.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
status.write_text(json.dumps(dict(state='launcher_started',pid=p.pid,unix_time=time.time(),output=str(out),physical_gpus=[2,3,4],envs_per_gpu=20480,total_frames=4000000000),indent=2))
print('LAUNCHED',p.pid,flush=True)
