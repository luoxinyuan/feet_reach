"""Run metric jobs concurrently; stream only selected representative videos."""
import argparse,os,subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.compliance_eval.report import build
from scripts.compliance_eval.protocol import digest
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,default=ROOT/'cfg/compliance_eval_v1_1.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0,1,5,6');p.add_argument('--videos',action='store_true');p.add_argument('--video-only',action='store_true');p.add_argument('--supervised-deltaq-checkpoint',type=Path);p.add_argument('--guidance-cache',type=Path);p.add_argument('--guidance-cache-map',type=Path);p.add_argument('--offline-prompt-cache-map',type=Path);p.add_argument('--prompt-only-behavior',action='store_true',help='Supply behavior force tokens to OfflineForcePromptCommand without dispatching physical wrenches');p.add_argument('--policy-branch',choices=['deploy','teacher'],default='deploy',help='Evaluate deployed student branch or privileged teacher actor');a=p.parse_args()
checkpoint_hash=digest(a.checkpoint);manifest_hash=digest(a.manifest)
supervised_hash=digest(a.supervised_deltaq_checkpoint) if a.supervised_deltaq_checkpoint else None
a.output.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',');started=time.monotonic()
base=[sys.executable,str(ROOT/'scripts/compliance_eval/worker.py'),'--checkpoint',str(a.checkpoint.resolve()),'--manifest',str(a.manifest.resolve())]
base += ['--policy-branch',a.policy_branch]
if a.guidance_cache:base+=['--guidance-cache',str(a.guidance_cache.resolve())]
if a.supervised_deltaq_checkpoint:
    base += ['--supervised-deltaq-checkpoint',str(a.supervised_deltaq_checkpoint.resolve())]
jobs=[]
if not a.video_only:
    jobs=[(f'tracking_seed{i}',['--suite','tracking','--seed-index',str(i)]) for i in range(3)]+[('behavior',['--suite','behavior']),('behavior_zero',['--suite','behavior','--zero'])]
    if a.prompt_only_behavior:
        jobs=[(name,args+(['--prompt-only-behavior'] if name in ('behavior','behavior_zero') else [])) for name,args in jobs]
if a.videos or a.video_only:
    m=json.loads(a.manifest.read_text());clips=m['clips'];chosen=[]
    for category in ['left_arm','locomotion','crouch']:
        candidates=[(i,r) for i,r in enumerate(clips) if r['split']=='validation' and r['category']==category]
        if candidates:chosen.append(sorted(candidates,key=lambda x:x[1]['frames'])[len(candidates)//2][0])
    jobs += [(f'video_tracking_{i}',['--suite','tracking','--video-clip',str(i)]) for i in chosen]
    jobs += [(f'video_{c}',['--suite','behavior','--video-case',c]) for c in ['left_ee_forward','root_forward','crouch','yaw_left','bend_forward']]
running={};pending=list(jobs);finished=[]
while pending or running:
    for gpu in gpus:
        if gpu in running or not pending:continue
        name,args=pending.pop(0);out=a.output/'metrics'/name
        if (name.startswith('tracking_') or name.startswith('video_tracking_')) and a.guidance_cache_map:
            args=args+['--guidance-cache-map',str(a.guidance_cache_map.resolve())]
        if (name.startswith('tracking_') or name.startswith('video_tracking_')) and a.offline_prompt_cache_map:
            args=args+['--offline-prompt-cache-map',str(a.offline_prompt_cache_map.resolve())]
        (a.output/'logs').mkdir(parents=True,exist_ok=True)
        if name.startswith('video_'):
            args=args+['--video-output',str((a.output/'videos'/(name.removeprefix('video_')+'.mp4')).resolve())]
        if (out/'result.json').exists():
            saved=json.loads((out/'result.json').read_text())
            if saved['checkpoint_sha256']!=checkpoint_hash or saved['manifest_sha256']!=manifest_hash:
                raise ValueError(f'Resume identity mismatch: {out}')
            if saved.get('policy_branch','deploy')!=a.policy_branch:
                raise ValueError(f'Policy branch identity mismatch: {out}')
            if saved.get('supervised_deltaq_checkpoint_sha256')!=supervised_hash:
                raise ValueError(f'Supervised model identity mismatch: {out}')
            if name.startswith('tracking_') or name.startswith('video_tracking_'):
                expected_cache_map_hash=digest(a.guidance_cache_map) if a.guidance_cache_map else None
                if saved.get('guidance_cache_map_sha256')!=expected_cache_map_hash:
                    raise ValueError(f'Guidance-cache mapping identity mismatch: {out}')
                expected_prompt_map_hash=digest(a.offline_prompt_cache_map) if a.offline_prompt_cache_map else None
                if saved.get('offline_prompt_cache_map_sha256')!=expected_prompt_map_hash:
                    raise ValueError(f'Offline prompt-cache mapping identity mismatch: {out}')
            finished.append(dict(name=name,cached=True));continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',PYTHONPATH=f'/home/xl521/IsaacLab/source/isaaclab:{ROOT}'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''),PYTHONUNBUFFERED='1')
        log=open(a.output/'logs'/(name+'.log'),'w');proc=subprocess.Popen(base+args+['--output',str(out.resolve())],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        running[gpu]=(name,proc,log,time.monotonic());print('START',name,'GPU',gpu,flush=True)
    for gpu,(name,proc,log,t0) in list(running.items()):
        if proc.poll() is None:continue
        log.close();record=dict(name=name,gpu=gpu,exit_code=proc.returncode,seconds=time.monotonic()-t0);finished.append(record);del running[gpu];print('DONE',record,flush=True)
        if proc.returncode:
            for _,other,_,_ in running.values():other.terminate()
            raise RuntimeError(f'{name} failed; see {a.output/"logs"/(name+".log")}')
    time.sleep(1)
summary=build(a.output);timing=dict(wall_seconds=time.monotonic()-started,jobs=finished,video_bytes=sum(p.stat().st_size for p in (a.output/'videos').glob('*.mp4')))
(a.output/('video_timing.json' if a.video_only else 'timing.json')).write_text(json.dumps(timing,indent=2));print(json.dumps(timing,indent=2))
