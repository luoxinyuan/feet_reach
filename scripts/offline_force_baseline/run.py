"""Reproducible reference-pose force encoding audit; does not train a policy."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from active_adaptation.utils.segmented_guidance import geometry_jacobians
from scripts.compliance_eval.protocol import load_arrays
from scripts.offline_force_baseline.solver import solve_sequence

def layouts(meta,geometry):
    excluded={'pelvis','head_mimic','left_hand_mimic','right_hand_mimic'}
    base=[dict(name=b,body=b,offset=[0,0,0]) for b in meta['body_names'] if b not in excluded]
    joints=[dict(name=e['child'],body=e['child'],offset=[0,0,0]) for e in geometry['edges'] if e['axis'] is not None]
    extended=list(joints)
    for body in ['left_hand_mimic','right_hand_mimic','left_ankle_roll_link','right_ankle_roll_link']:
        for i,off in enumerate([[.05,0,0],[0,.05,0]]):extended.append(dict(name=body+f'_offset{i}',body=body,offset=off))
    return dict(origins23=base,origins29=joints,offsets37=extended)

def metrics(A,dq,f,k=80.):
    pred=np.einsum('tnk,tk->tn',A,f.reshape(len(f),-1))/k;e=pred-dq;norms=np.linalg.norm(f,axis=-1)
    return dict(rmse_rad=float(np.sqrt(np.mean(e*e))),mae_rad=float(abs(e).mean()),relative_error=float(np.linalg.norm(e)/max(np.linalg.norm(dq),1e-12)),
        joint_rmse_rad=np.sqrt(np.mean(e*e,axis=0)).tolist(),error_sse=float(np.sum(e*e)),target_sse=float(np.sum(dq*dq)),scalar_count=int(dq.size),
        total_force_mean_N=float(norms.sum(1).mean()),total_force_max_N=float(norms.sum(1).max()),point_max_N=float(norms.max()),
        vector_net_mean_N=float(np.linalg.norm(f.sum(1),axis=-1).mean()),active_points_mean=float((norms>.05).sum(1).mean()),
        slew_max_N_s=float(np.linalg.norm(np.diff(f,axis=0),axis=-1).max()/.02))

def process(job):
    record,layout,points,out,budgets,frames,k=job;out=Path(out);dest=out/'clips'/f"{record['id']:02d}_{layout}.json"
    if dest.exists():return json.loads(dest.read_text())
    meta,a=load_arrays(ROOT/'dataset'/record['dataset']);st=record['start'];T=record['frames']
    q=np.asarray(a['joint_pos'][st:st+T+1],float);rp=np.asarray(a['root_pos_w'][st:st+T],float);rq=np.asarray(a['root_quat_w'][st:st+T],float)
    geom=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
    pos,j=geometry_jacobians(q[:-1],rp,rq,meta['joint_names'],points,geom)
    A=j[:,:,:,6:].transpose(0,3,1,2).reshape(T,q.shape[1],-1);dq=np.diff(q,axis=0)
    U,s,Vh=np.linalg.svd(A,full_matrices=False);tol=np.maximum(s[:,:1]*1e-7,1e-10);valid=s>tol
    inv=np.divide(1,s,out=np.zeros_like(s),where=valid)
    f=np.einsum('tkd,tk->td',Vh,inv*np.einsum('tnk,tn->tk',U,k*dq)).reshape(T,len(points),3)
    rank=valid.sum(1)
    result=dict(record=record,layout=layout,points=points,joint_names=meta['joint_names'],rank_min=int(rank.min()),rank_max=int(rank.max()),rank_mean=float(rank.mean()),
                singular_values_median=np.median(s,axis=0).tolist(),joint_row_norm_median=np.median(np.linalg.norm(A,axis=2),axis=0).tolist(),force_noise_0_1N_decode_rmse_rad=float(.1*np.sqrt(np.mean(np.sum(A*A,axis=2)))/k),methods={})
    result['methods']['pinv']=metrics(A,dq,f,k)
    cache=dict(q=q,root_pos=rp,root_quat=rq,positions=pos,dq=dq,singular_values=s,pinv=f)
    for budget in budgets:
        for smooth in [False,True]:
            name=f'budget{budget:g}'+('_smooth' if smooth else '')
            force,stats=solve_sequence(A,dq,budget,k=k,slew=150 if smooth else None,smooth=.01 if smooth else 0.)
            result['methods'][name]=dict(**metrics(A,dq,force,k),solver=stats);cache[name]=force.astype(np.float32)
    np.savez_compressed(dest.with_suffix('.npz'),**cache)
    dest.write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=64);p.add_argument('--workers',type=int,default=4);p.add_argument('--budgets',type=float,nargs='+',default=[10,20,40,60]);p.add_argument('--limit',type=int);p.add_argument('--per-category',type=int,default=3);p.add_argument('--gain',type=float,default=80.);p.add_argument('--layouts',nargs='+',default=['origins23','origins29','offsets37']);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'clips').mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'cfg/compliance_eval_v1_1.json').read_text());records=[];seen={}
    for row in manifest['clips']:
        key=(row['split'],row['category'])
        if seen.get(key,0)>=args.per_category:continue
        seen[key]=seen.get(key,0)+1;meta,a=load_arrays(ROOT/'dataset'/row['dataset']);lo=int(meta['starts'][row['clip']]);hi=int(meta['ends'][row['clip']]);T=min(args.frames,hi-lo-1)
        start=lo+max(0,(min(hi-lo,1000)-T-1)//2)
        records.append(dict(row,id=len(records),start=start,frames=T))
    if args.limit:records=records[:args.limit]
    meta,_=load_arrays(ROOT/'dataset'/records[0]['dataset']);geometry=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text());ls={k:v for k,v in layouts(meta,geometry).items() if k in args.layouts}
    config=dict(records=records,layouts=ls,budgets=args.budgets,K=args.gain,dt=.02,point_cap_N=30,slew_N_s=150,frames=args.frames,error_slack_rad=1e-4,solver='clarabel 0.11.1',
                note='Offsets are virtual body-attached diagnostic points; surface/contact accessibility not validated. Only joint delta is encoded; floating-base translation/yaw are not target coordinates.')
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Output manifest mismatch')
    path.write_text(json.dumps(config,indent=2));start=time.monotonic();results=[]
    jobs=[(r,name,points,str(args.output),args.budgets,args.frames,args.gain) for r in records for name,points in ls.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut=[pool.submit(process,j) for j in jobs]
        for f in as_completed(fut):
            r=f.result();results.append(r);print(f"DONE {len(results)}/{len(jobs)} {r['record']['split']} {r['record']['category']} {r['layout']} rank={r['rank_min']}-{r['rank_max']} elapsed={time.monotonic()-start:.1f}s",flush=True)
    summary={}
    for split in ['train_diagnostic','validation']:
        for layout in ls:
            subset=[r for r in results if r['layout']==layout and r['record']['split']==split]
            if not subset:continue
            for method in subset[0]['methods']:
                ms=[r['methods'][method] for r in subset];sse=sum(x['error_sse'] for x in ms);signal=sum(x['target_sse'] for x in ms)
                summary[f'{split}/{layout}/{method}']=dict(clips=len(ms),rmse_rad=float(np.sqrt(sse/sum(x['scalar_count'] for x in ms))),relative_error=float(np.sqrt(sse/max(signal,1e-12))),
                    total_force_mean_N=float(np.mean([x['total_force_mean_N'] for x in ms])),total_force_max_N=max(x['total_force_max_N'] for x in ms),point_max_N=max(x['point_max_N'] for x in ms),slew_max_N_s=max(x['slew_max_N_s'] for x in ms))
    (args.output/'summary.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,metrics=summary),indent=2))
if __name__=='__main__':main()
