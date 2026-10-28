"""Precompute deterministic trajectory guidance; resumable per-clip output."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.segmented_guidance import geometry_jacobians,nominal_target,weighted_problem,select_segments,optimize_forces
from scripts.data_process.prepare_single_hand_ee_guidance import direct_profile

def initialize(split,out,point_count,points_config=None,source_dataset=None,single_hand_source=None):
    global DS,OUT,POINTS,GEOM,ALLOWED_RANGES,SOURCE_DATASET,SOURCE_REPORTS,SINGLE_HAND_SOURCE
    torch.set_num_threads(1)
    SOURCE_DATASET=Path(source_dataset).resolve() if source_dataset else ROOT/f'dataset/passive_whole_body_full/{split}'
    SINGLE_HAND_SOURCE=Path(single_hand_source).resolve() if single_hand_source else ROOT/'artifacts/ceer_single_hand_ee/full_20261025'
    DS=MotionDataset.create_from_path_lazy(str(SOURCE_DATASET))
    OUT=Path(out)
    points_file=Path(points_config).resolve() if points_config else ROOT/'cfg/task/G1/G1_ceer_spring18_compliance.yaml'
    POINTS=yaml.safe_load(points_file.read_text())['command']['force_points'][:point_count]
    GEOM=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
    meta=json.loads((SOURCE_DATASET/'meta_motion.json').read_text())
    allowed={}
    for lo,hi in zip(meta['starts'],meta['ends']):
        lo,hi=int(lo),int(hi)
        mid=int(DS.data.motion_id[lo]);step0=int(DS.data.step[lo]);step1=int(DS.data.step[hi-1])+1
        allowed.setdefault(mid,[]).append((step0,step1))
    ALLOWED_RANGES=allowed
    manifest=SOURCE_DATASET.parent/'manifest.json'
    SOURCE_REPORTS={int(r['motion_id']):r for r in json.loads(manifest.read_text()).get('reports',[])} if manifest.exists() else {}
    # Process complete clips so force indexing remains by raw motion_id and step.
    mids=DS.data.motion_id
    changes=torch.where(mids[1:]!=mids[:-1])[0].tolist()
    starts=[0]+[int(i)+1 for i in changes];ends=[int(i)+1 for i in changes]+[len(mids)]
    DS.starts=torch.as_tensor(starts,dtype=torch.int32);DS.ends=torch.as_tensor(ends,dtype=torch.int32);DS.lengths=DS.ends-DS.starts

def process(i):
    dest=OUT/'clips'/f'{i:05d}.npz'
    if dest.exists():return json.loads(str(np.load(dest)['report']))
    st,en=int(DS.starts[i]),int(DS.ends[i]);m=DS.data[st:en]
    mid=int(m.motion_id[0]);assert bool((m.motion_id==mid).all())
    allowed_ranges=ALLOWED_RANGES.get(mid,[])
    source_report=SOURCE_REPORTS.get(mid,{})
    q=m.joint_pos.numpy().astype(float);n=len(q)
    zero_profile=str(source_report.get('profile','')).startswith('static hold')
    single_hand='side' in source_report and 'index' in source_report
    if single_hand and not zero_profile:
        record=dict(index=int(source_report['index']),file=source_report['input_file'])
        _,slots,force=direct_profile(record,str(SINGLE_HAND_SOURCE),POINTS,20261028)
        residual=np.zeros(n,dtype=np.float32);arm_errors=[np.zeros(n),np.zeros(n)]
        stats={};qp={};accepted=[(max(0,a),min(n,b)) for a,b in allowed_ranges if min(n,b)-max(0,a)>=50]
    elif zero_profile:
        slots=np.full((n,3),-1,dtype=np.int16);force=np.zeros((n,3,3),dtype=np.float32)
        residual=np.zeros(n,dtype=np.float32);arm_errors=[np.zeros(n),np.zeros(n)]
        stats={};qp={};accepted=[(max(0,a),min(n,b)) for a,b in allowed_ranges if min(n,b)-max(0,a)>=50]
    else:
        rp=m.root_pos_w.numpy().astype(float);rq=m.root_quat_w.numpy().astype(float)
        _,jac=geometry_jacobians(q,rp,rq,DS.joint_names,POINTS,GEOM)
        target=nominal_target(q,rp,rq);a,b,groups,weights=weighted_problem(jac,target,DS.joint_names)
        slots,stats=select_segments(a,b,len(POINTS))
        force,residual,qp=optimize_forces(a,b,slots)
        dense=np.zeros((n,len(POINTS),3))
        for k in range(3):
            mask=slots[:,k]>=0;dense[np.where(mask)[0],slots[mask,k]]=force[mask,k]
        actual=np.einsum('tpjd,tpj->td',jac,dense)
        arm_errors=[]
        for ids in groups[3:]:
            err=np.linalg.norm(actual[:,ids]-target[:,ids],axis=1)/np.maximum(np.linalg.norm(target[:,ids],axis=1),.3)
            arm_errors.append(err)
        boundaries=np.r_[0,np.where(np.any(slots[1:]!=slots[:-1],axis=1))[0]+1,n]
        accepted=[]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            if residual[lo:hi].mean()<=.85 and max(e[lo:hi].mean() for e in arm_errors)<=.85:
                for a0,b0 in allowed_ranges:
                    a1,b1=max(int(lo),a0),min(int(hi),b0)
                    if b1-a1>=50:accepted.append((a1,b1))
    # Merge adjacent accepted segments; short isolated spans are not useful training episodes.
    merged=[]
    for lo,hi in accepted:
        if merged and merged[-1][1]==lo:merged[-1][1]=hi
        else:merged.append([lo,hi])
    merged=[s for s in merged if s[1]-s[0]>=50]
    steps=m.step.numpy();assert np.array_equal(steps,np.arange(len(q)))
    counts=np.bincount(slots[slots>=0],minlength=len(POINTS)).tolist()
    report=dict(source_report,clip=i,motion_id=mid,frames=len(q),start=st,accepted_spans=merged,
        accepted_frames=sum(hi-lo for lo,hi in merged),mean_relative_error=float(residual.mean()),
        arm_relative_error=[float(e.mean()) for e in arm_errors],point_frames=counts,**stats,**qp)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez(tmp,slots=slots,forces=force,residual=residual.astype(np.float32),report=json.dumps(report))
    tmp.replace(dest)
    return report

def assemble(out,split,reports):
    out=Path(out);lengths=[r['frames'] for r in reports];offset=np.r_[0,np.cumsum(lengths)]
    total=int(offset[-1]);slots=np.lib.format.open_memmap(out/'slots.npy',mode='w+',dtype=np.int16,shape=(total,3))
    forces=np.lib.format.open_memmap(out/'forces.npy',mode='w+',dtype=np.float32,shape=(total,3,3))
    lookup=np.full(max(r['motion_id'] for r in reports)+1,-1,np.int64)
    source=SOURCE_DATASET
    meta=json.loads((source/'meta_motion.json').read_text());starts=[];ends=[]
    for i,r in enumerate(reports):
        z=np.load(out/'clips'/f'{r["clip"]:05d}.npz');slots[offset[i]:offset[i+1]]=z['slots'];forces[offset[i]:offset[i+1]]=z['forces'];lookup[r['motion_id']]=offset[i]
        for lo,hi in r['accepted_spans']:starts.append(r['start']+lo);ends.append(r['start']+hi)
    slots.flush();forces.flush();np.save(out/'motion_offsets.npy',lookup)
    view=out/'dataset';view.mkdir(exist_ok=True)
    for name in ('meta.json','_tensordict'):
        p=view/name
        if not p.exists():p.symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
    meta.update(starts=starts,ends=ends,info={});(view/'meta_motion.json').write_text(json.dumps(meta))
    summary=dict(split=split,clips=len(reports),frames=total,accepted_frames=sum(r['accepted_frames'] for r in reports),
        accepted_spans=len(starts),mean_relative_error=float(np.average([r['mean_relative_error'] for r in reports],weights=lengths)),
        point_frames=np.sum([r['point_frames'] for r in reports],axis=0).tolist(),
        points=POINTS,geometry_sha256=GEOM['sha256'],reports=reports,
        constraints={'per_point_N':30,'total_N':60,'slew_N_per_s':150,'force_ball':'conservative L1','max_points':3},
        target='80 joint delta; reference-only root translation/yaw feedforward; no actual-state feedback')
    (out/'manifest.json').write_text(json.dumps(summary,indent=2))
    # Manifest is the completion marker. Cache config must match this exact point list.
    print('CACHE_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ('reports','points')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='train');p.add_argument('--out',required=True);p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=8);p.add_argument('--points',type=int,default=18);p.add_argument('--points-config');p.add_argument('--source-dataset');p.add_argument('--indices',default='')
    p.add_argument('--single-hand-source')
    args=p.parse_args();out=Path(args.out).resolve();(out/'clips').mkdir(parents=True,exist_ok=True)
    initialize(args.split,out,args.points,args.points_config,args.source_dataset,args.single_hand_source)
    ids=list(range(DS.num_motions))
    if args.indices:ids=[int(x) for x in args.indices.split(',')]
    elif args.limit:ids=np.linspace(0,len(ids)-1,args.limit,dtype=int).tolist()
    reports=[];start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.split,str(out),args.points,args.points_config,args.source_dataset,args.single_hand_source)) as pool:
        for report in pool.map(process,ids,chunksize=1):
            reports.append(report)
            if len(reports)%10==0 or len(reports)==len(ids):print('PROGRESS',len(reports),len(ids),'seconds',round(time.time()-start,1),'accepted',round(sum(r['accepted_frames'] for r in reports)/sum(r['frames'] for r in reports),3),flush=True)
    assemble(out,args.split,reports)
