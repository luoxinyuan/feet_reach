"""Replay surface spring variants on the frozen v2 representative split."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from active_adaptation.utils.surface_guidance import SurfaceSampler
from active_adaptation.utils.contact_observability import greedy_contacts
from active_adaptation.utils.segmented_guidance import geometry_jacobians,rot
from active_adaptation.utils.body_guidance import slew_force,limit_force
from scripts.compliance_eval.protocol import load_arrays

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pool',type=int,default=64);p.add_argument('--upper-body',action='store_true');p.add_argument('--coverage-prior',type=float,default=0.);p.add_argument('--full-span',action='store_true');p.add_argument('--covariance',choices=['none','global','conditional'],default='none');p.add_argument('--random5',action='store_true',help='Include a uniform five-contact baseline from the same valid candidate pool');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 source=ROOT/'artifacts/force_mapping_eval/20261111_v2_root_valid';base_file=np.load(source/'data.npz');base={k:base_file[k] for k in base_file.files};manifest=json.loads((source/'manifest.json').read_text());cache=json.loads((Path(manifest['force_cache'])/'manifest.json').read_text());arrays=[load_arrays(Path(x)) for x in cache['dataset_paths']]
 geom=cache['geometry'];names=manifest['joint_names'];bodies=['pelvis']+[e['child'] for e in geom['edges']];surface=SurfaceSampler(ROOT/'artifacts/surface_compliance/geometry',bodies,'cpu')
 covariance=None
 if a.covariance!='none':
  tr=base['split']==0;D=np.c_[base['delta_root_1'],base['delta_q']][tr].astype(np.float64)
  if a.covariance=='conditional':
   X=base['state'][tr].astype(np.float64);X=np.c_[(X-X.mean(0))/np.maximum(X.std(0),.01),np.ones(len(X))];D=D-X@np.linalg.solve(X.T@X+.01*np.eye(X.shape[1]),X.T@D)
  C=D.T@D/len(D);C/=np.trace(C)/len(C);C+=.001*np.eye(len(C));covariance=torch.tensor(C,dtype=torch.float32)
  (a.output/'covariance.json').write_text(json.dumps(dict(joint_names=names,matrix=C.tolist(),fit_split='train_only',kind=a.covariance)))
 # Equal-dimensional canonical 5-contact interface for all methods.
 # Local moment preserves lever-arm information without conveying unloaded contact positions.
 methods=['random3','greedy3','greedy5']
 if a.random5:methods.append('random5')
 features={k:[] for k in methods};stats={k:[] for k in methods};analytic={k:[] for k in methods};started=time.time()
 def tokens(body,local,f,R):
  visible=np.linalg.norm(f,axis=-1)>1e-5;body=body[visible];local=local[visible];f=f[visible]@R
  order=np.lexsort((local[:,2],local[:,1],local[:,0],body))
  token=np.zeros((5,len(bodies)+7),np.float32)
  for j,i in enumerate(order):
   if j>=5:raise ValueError('More than five simultaneous forces')
   token[j,body[i]]=1;token[j,len(bodies):len(bodies)+3]=local[i];token[j,len(bodies)+3:len(bodies)+6]=f[i];token[j,-1]=1
  return token.flatten()
 for ri,c in enumerate(manifest['records']):
  torch.manual_seed(20261114+c['eval_id']);meta,data=arrays[c['source_index']];T=c['eval_frames'];start=c['start'];available=min(c['frames'],T+31);q=np.array(data['joint_pos'][start:start+available]);rp=np.array(data['root_pos_w'][start:start+available]);rq=np.array(data['root_quat_w'][start:start+available]);R=rot(rq)
  # Full body origin orientations reconstructed by FK from three basis points.
  origin_points=[dict(body=b,offset=[0,0,0]) for b in bodies]

  out={k:np.zeros((T,5*(len(bodies)+7)),np.float32) for k in methods};norms={k:[] for k in methods};err={k:[] for k in methods}
  # Fixed 0.8s contact windows, final 0.1s unloading; causal application lag.
  for st in range(0,T,40):
   en=min(st+40,T);body,tri,bary=surface.sample((a.pool,));local=(tri*bary[...,None]).sum(-2).numpy();body=body.numpy();points=[dict(body=bodies[b],offset=x.tolist()) for b,x in zip(body,local)]
   pos,jac=geometry_jacobians(q[st:en+1],rp[st:en+1],rq[st:en+1],names,points,geom)
   good=pos[0,:,2]>.06
   if a.upper_body:good &= np.array([not any(x in bodies[b] for x in ("hip","knee","ankle")) for b in body])
   jac[0,~good]=0
   # Match production clamping of each lookahead FRAME at the true clip end,
   # rather than repeating the last nonzero displacement of the evaluation crop.
   hs=np.asarray([0,10,20,30] if a.full_span else [0,4,8]);t0=np.minimum(st+hs,len(q)-1);t1=np.minimum(t0+1,len(q)-1)
   delta=np.c_[rp[t1]-rp[t0],(Rotation.from_matrix(R[t1])*Rotation.from_matrix(R[t0]).inv()).as_rotvec(),q[t1]-q[t0]].astype(np.float32)
   hs=np.arange(len(hs))
   # Covariance is learned in current-root coordinates, so transform the first
   # six generalized columns and displacement vectors into that same frame.
   select_jac=jac[0:1].copy();select_delta=delta[hs].copy()
   if covariance is not None:
    select_jac[...,:3]=select_jac[...,:3]@R[st];select_jac[...,3:6]=select_jac[...,3:6]@R[st]
    select_delta[:,:3]=select_delta[:,:3]@R[st];select_delta[:,3:6]=select_delta[:,3:6]@R[st]

   coverage=torch.tensor(body[None,:,None]==np.array([bodies.index(b) for b in ['pelvis','left_wrist_yaw_link','right_wrist_yaw_link']])[None,None])
   selected,active=greedy_contacts(torch.tensor(select_jac,dtype=torch.float32),torch.tensor(select_delta[None]),5,covariance=covariance,coverage=coverage,coverage_penalty=a.coverage_prior)
   selections={'greedy5':(selected[0].numpy(),active[0].numpy())}
   s3,m3=greedy_contacts(torch.tensor(select_jac,dtype=torch.float32),torch.tensor(select_delta[None]),3,covariance=covariance,coverage=coverage,coverage_penalty=a.coverage_prior);selections['greedy3']=(s3[0].numpy(),m3[0].numpy())
   valid=np.flatnonzero(good);selections['random3']=(np.random.default_rng(c['eval_id']+st).choice(valid,3,replace=False),np.ones(3,bool))
   if a.random5:
    count=min(5,len(valid))
    picks=np.random.default_rng(c['eval_id']+st).choice(valid,count,replace=False)
    selections['random5']=(picks,np.ones(count,dtype=bool))
   for method,(ids,mask) in selections.items():
    ids=ids[mask];current=torch.zeros(1,len(ids),3);releasing=False
    for t in range(en-st):
     f=current[0].numpy();out[method][st+t]=tokens(body[ids],local[ids],f,R[st+t])
     norms[method].append([float(np.linalg.norm(f,axis=-1).sum()),float(np.linalg.norm(f,axis=-1).max()),int((np.linalg.norm(f,axis=-1)>1e-5).sum())])
     raw=torch.tensor(80*(pos[t+1,ids]-pos[t,ids]),dtype=torch.float32)[None]
     desired=limit_force(raw)
     releasing |= (40-t)*.02 <= float(current.norm(dim=-1).max())/150.+.040001
     if releasing:desired.zero_()
     # With 30N cap, 0.1s isn't always enough to unload. Continue hold/contact
     # until zero is essential; nominal forces here are low, assert before switching.
     for _ in range(4):current=slew_force(current,desired,.005)
    if en<T and current.norm()>1e-5:raise RuntimeError('Unload window insufficient; extend it, never teleport force')

  for k in methods:features[k].append(out[k].reshape(T,-1));stats[k].extend(norms[k])
  if (ri+1)%20==0:print('clips',ri+1,'seconds',round(time.time()-started),flush=True)
 # Same contact tokens and same decoder width for all three original baselines.
 for name in ['jacobian3','jacobian9','spring3']:
  bb=np.array([bodies.index(b) for b,x in manifest['canonical_points']]);xx=np.array([x for b,x in manifest['canonical_points']])
  # Baseline forces are already root-frame; local offsets need no transform.
  features[name]=np.stack([tokens(bb,xx,f,np.eye(3)) for f in base[name]])
 features.update({k:np.concatenate(v) for k,v in list(features.items()) if isinstance(v,list)})
 np.savez_compressed(a.output/'data.npz',**features,**{k:base[k] for k in ['state','delta_q','delta_root_1','delta_root_5','split','clip','step']})
 (a.output/'manifest.json').write_text(json.dumps(dict(source=str(source),pool=a.pool,coverage_prior=a.coverage_prior,full_span=a.full_span,covariance=a.covariance,interface="contact_tokens",upper_body=a.upper_body,adaptive_release=True,bodies=bodies,methods=list(features),seconds=time.time()-started,force_stats={k:dict(mean_total=float(np.mean(v,0)[0]),max_total=float(np.max(v,0)[0]),max_point=float(np.max(v,0)[1]),mean_count=float(np.mean(v,0)[2])) for k,v in stats.items()}),indent=2))
if __name__=='__main__':main()
