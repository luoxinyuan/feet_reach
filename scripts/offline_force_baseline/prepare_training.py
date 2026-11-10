"""Pack all selected training clips and solve reference-only 37-point prompts.

64-transition SOCP blocks carry the preceding force as a hard slew boundary.
No forces are computed from simulated states. Completed clips are resumable.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,hashlib,shutil,multiprocessing
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.compliance_eval.protocol import load_arrays
from scripts.offline_force_baseline.solver import solve_sequence
from active_adaptation.utils.segmented_guidance import geometry_jacobians

SOURCES=[ROOT/'artifacts/ceer_segmented_jacobian/20261027-ee-root3-multi10/dataset',ROOT/'dataset/crouch_lowlevel/train']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pack(out,limit=None):
    if (out/'plan.json').exists():return json.loads((out/'plan.json').read_text())
    geom=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
    ref=json.loads((ROOT/'artifacts/offline_force_baseline/20261105_calibrated_budget60/manifest.json').read_text())
    plan=dict(points=ref['layouts']['offsets37'],K=ref['K'],budget_N=60.,point_cap_N=30.,slew_N_s=150.,dt=.02,block=64,smooth=.01,error_slack_rad=1e-4,geometry=geom,sources=[],clips=[],dataset_paths=[],weights=[.8,.2],frames=0)
    mid=0;offset=0
    for si,source in enumerate(SOURCES):
        meta,arr=load_arrays(source);indices=list(range(len(meta['starts'])))
        if limit:indices=indices[:limit]
        dest=out/f'dataset{si}';td=dest/'_tensordict';td.mkdir(parents=True,exist_ok=True)
        lengths=[meta['ends'][i]-meta['starts'][i] for i in indices];total=sum(lengths)
        tmeta=json.loads((source/'_tensordict/meta.json').read_text());newmeta=dict(meta,starts=[],ends=[]);newmeta['info']={}
        packed={}
        for key,value in arr.items():
            shape=(total,*value.shape[1:]);packed[key]=np.memmap(td/f'{key}.memmap',mode='w+',dtype=value.dtype,shape=shape)
            tmeta[key]['shape']=list(shape)
        at=0
        for ci,length in zip(indices,lengths):
            lo=meta['starts'][ci];hi=meta['ends'][ci]
            for key,value in arr.items():packed[key][at:at+length]=value[lo:hi]
            packed['motion_id'][at:at+length]=mid;packed['step'][at:at+length]=np.arange(length)
            newmeta['starts'].append(at);newmeta['ends'].append(at+length)
            plan['clips'].append(dict(id=mid,source_index=si,source_clip=ci,start=at,frames=length,force_offset=offset,source_start=lo))
            at+=length;offset+=length;mid+=1
        for value in packed.values():value.flush()
        tmeta['shape']=[total];(td/'meta.json').write_text(json.dumps(tmeta));shutil.copyfile(source/'meta.json',dest/'meta.json');(dest/'meta_motion.json').write_text(json.dumps(newmeta))
        plan['sources'].append(dict(path=str(source),meta_sha256=sha(source/'meta_motion.json')));plan['dataset_paths'].append(str(dest.resolve()))
        print('PACKED',si,total,len(indices),flush=True)
    plan['frames']=offset;plan['joint_names']=meta['joint_names'];plan['dataset_meta_sha256']=[sha(Path(p)/'meta_motion.json') for p in plan['dataset_paths']]
    np.save(out/'motion_offsets.npy',np.array([c['force_offset'] for c in plan['clips']],dtype=np.int64))
    f=np.lib.format.open_memmap(out/'forces.npy',mode='w+',dtype=np.float32,shape=(offset,37,3));f[:]=np.nan;f.flush()
    (out/'plan.json').write_text(json.dumps(plan,indent=2));return plan

G={}
def init_worker(out):
    out=Path(out);p=json.loads((out/'plan.json').read_text());G.update(out=out,plan=p,arrays=[load_arrays(Path(x))[1] for x in p['dataset_paths']],forces=np.load(out/'forces.npy',mmap_mode='r+'))

def solve_clip(c):
    out=G['out'];p=G['plan'];done=out/'stats'/f"{c['id']:05d}.json"
    if done.exists():return json.loads(done.read_text())
    a=G['arrays'][c['source_index']];start=c['start'];length=c['frames'];off=c['force_offset'];q=np.asarray(a['joint_pos'][start:start+length],float)
    prev=None;stats=[];sse=signal=0.;max_total=0.;max_slew=0.;total=0.
    for lo in range(0,length-1,p['block']):
        hi=min(length-1,lo+p['block'])
        _,j=geometry_jacobians(q[lo:hi],np.asarray(a['root_pos_w'][start+lo:start+hi],float),np.asarray(a['root_quat_w'][start+lo:start+hi],float),p['joint_names'],p['points'],p['geometry'])
        A=j[:,:,:,6:].transpose(0,3,1,2).reshape(hi-lo,29,111);dq=q[lo+1:hi+1]-q[lo:hi]
        try:f,s=solve_sequence(A,dq,60,k=p['K'],slew=150,smooth=.01,previous_force=prev)
        except Exception as e:raise RuntimeError(f'clip={c["id"]} block={lo}:{hi}: {e}') from e
        err=np.einsum('tnk,tk->tn',A,f.reshape(hi-lo,-1))/p['K']-dq
        sse+=float((err*err).sum());signal+=float((dq*dq).sum());norm=np.linalg.norm(f,axis=-1).sum(1);max_total=max(max_total,float(norm.max()));total+=float(norm.sum())
        diffs=np.diff(f,axis=0) if prev is None else np.diff(np.concatenate([prev[None],f]),axis=0)
        max_slew=max(max_slew,float(np.linalg.norm(diffs,axis=-1).max(initial=0)/.02));prev=f[-1]
        G['forces'][off+lo:off+hi]=f;stats.append(s)
    # Last frame has no next transition and is never stepped past; use zero prompt.
    G['forces'][off+length-1]=0
    result=dict(id=c['id'],transitions=length-1,error_sse=sse,target_sse=signal,total_force_mean_N=total/max(1,length-1),total_force_max_N=max_total,slew_max_N_s=max_slew,solver=stats)
    tmp=done.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(done);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=96);p.add_argument('--limit',type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'stats').mkdir(exist_ok=True)
    plan=pack(a.output,a.limit);start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=init_worker,initargs=(str(a.output),)) as pool:
        futures=[pool.submit(solve_clip,c) for c in plan['clips']]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%100==0 or len(results)==len(futures):print('SOLVED',len(results),'/',len(futures),'seconds',round(time.time()-start,1),flush=True)
    forces=np.load(a.output/'forces.npy',mmap_mode='r');assert np.isfinite(forces).all()
    summary=dict(clips=len(results),frames=plan['frames'],transitions=sum(r['transitions'] for r in results),relative_error=float(np.sqrt(sum(r['error_sse'] for r in results)/max(1e-20,sum(r['target_sse'] for r in results)))),max_total_N=max(r['total_force_max_N'] for r in results),max_slew_N_s=max(r['slew_max_N_s'] for r in results),solve_wall_seconds=time.time()-start)
    plan['summary']=summary;plan['complete']=True
    tmp=a.output/'manifest.tmp';tmp.write_text(json.dumps(plan,indent=2));tmp.replace(a.output/'manifest.json');print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
