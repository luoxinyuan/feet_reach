"""Generate full train/val dense force caches, fit K on train, audit aggregate loads."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.dense_link_guidance import link_names, displacement, cap_vectors, unit_metrics, describe, choose_k

def prepare(source,out,names,k=None,calibration=None):
    ds=MotionDataset.create_from_path_lazy(str(source.resolve()))
    ids=[ds.body_names.index(n) for n in names];n=len(ds.data)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError(f'Refusing to overwrite completed cache: {out}')
    mids=ds.data.motion_id.numpy().reshape(-1);steps=ds.data.step.numpy().reshape(-1)
    selected=np.zeros(n,dtype=bool);valid=np.zeros(n,dtype=bool)
    for lo,hi in zip(ds.starts.tolist(),ds.ends.tolist()):
        selected[lo:hi]=True;valid[lo:hi-1]=True
    valid[:-1]&=(mids[1:]==mids[:-1])&(steps[1:]==steps[:-1]+1);valid[-1]=False
    offsets=np.full(int(mids.max())+1,-1,dtype=np.int64)
    for lo in np.r_[0,np.where(mids[1:]!=mids[:-1])[0]+1]:
        mid=int(mids[lo]);value=int(lo-steps[lo])
        if offsets[mid]>=0 and offsets[mid]!=value:raise ValueError('Non-contiguous original motion ID')
        offsets[mid]=value
    if not np.array_equal(offsets[mids]+steps,np.arange(n)):raise ValueError('Cache indexing mismatch')
    cache=np.lib.format.open_memmap(out/'forces.npy',mode='w+',dtype=np.float32,shape=(n,len(names),3))
    metrics={key:np.empty(n,dtype=np.float32) for key in ['point_max','total','net','moment','yaw']}
    for lo in range(0,n,16384):
        hi=min(n,lo+16384);stop=min(hi+1,n)
        pos=ds.data.body_pos_w[lo:stop,ids].float().numpy()
        if hi==n:pos=np.concatenate([pos,pos[-1:]],axis=0)
        delta=displacement(pos,valid[lo:hi]);cache[lo:hi]=delta
        root=ds.data.root_pos_w[lo:hi].float().numpy()
        for key,v in unit_metrics(delta,pos[:-1]-root[:,None]).items():metrics[key][lo:hi]=v
    used={key:v[selected] for key,v in metrics.items()}
    if k is None:
        k,calibration=choose_k(used)
    unit={key:describe(v) for key,v in used.items()}
    max_clip_error=0.;slew=[];active=[]
    for lo in range(0,n,16384):
        hi=min(n,lo+16384);raw=np.array(cache[lo:hi])*k;force=cap_vectors(raw)
        max_clip_error=max(max_clip_error,float(np.linalg.norm(raw-force,axis=-1).max()))
        cache[lo:hi]=force
        pos=ds.data.body_pos_w[lo:hi,ids].float().numpy();root=ds.data.root_pos_w[lo:hi].float().numpy()
        for key,v in unit_metrics(force,pos-root[:,None]).items():metrics[key][lo:hi]=v
        active.append((np.linalg.norm(force,axis=-1)>1e-5).sum(-1)[selected[lo:hi]])
    cache.flush()
    for lo in range(0,n-1,16384):
        hi=min(n-1,lo+16384);v=np.linalg.norm(cache[lo+1:hi+1]-cache[lo:hi],axis=-1).max(-1)/.02
        slew.append(v[valid[lo:hi]])
    np.save(out/'motion_offsets.npy',offsets);np.save(out/'valid_next.npy',valid)
    summary=dict(schema=1,source_dataset=str(source.resolve()),source_meta_sha256=hashlib.sha256((source/'meta_motion.json').read_bytes()).hexdigest(),
        body_names=names,dt=.02,frames=n,selected_frames=int(selected.sum()),clips=ds.num_motions,
        k_N_per_m=float(k),calibration_train_limits=calibration,
        budget_policy=dict(per_point_N=30,total_p99_N=60,total_max_N=120,net_max_N=80),
        formula='F[t,i] = clip_norm(K*(x_ref[t+1,i]-x_ref[t,i]),30); world frame; zero at metadata clip ends',
        unit_displacement_metrics=unit,force_metrics={key:describe(v[selected]) for key,v in metrics.items()},
        active_points=describe(np.concatenate(active)),nominal_slew_N_per_s=describe(np.concatenate(slew)),max_clipping_error_N=max_clip_error,
        files_sha256={})
    for name in ['forces.npy','motion_offsets.npy','valid_next.npy']:
        digest=hashlib.sha256()
        with (out/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
        summary['files_sha256'][name]=digest.hexdigest()
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'dataset/passive_whole_body_full');p.add_argument('--output',type=Path,default=ROOT/'artifacts/ceer_dense_link/full23');p.add_argument('--include-mimic',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    meta=json.loads((a.dataset/'train/meta_motion.json').read_text());names=link_names(meta['body_names'],a.include_mimic)
    train=prepare(a.dataset/'train',a.output/'train',names)
    val=prepare(a.dataset/'val',a.output/'val',names,train['k_N_per_m'],train['calibration_train_limits'])
    report=dict(train=train,val=val);(a.output/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({split:{k:r[k] for k in ['k_N_per_m','force_metrics','active_points','nominal_slew_N_per_s']} for split,r in report.items()},indent=2),flush=True)
if __name__=='__main__':main()
