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
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pool',type=int,default=64);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 source=ROOT/'artifacts/force_mapping_eval/20261111_v2_root_valid';base_file=np.load(source/'data.npz');base={k:base_file[k] for k in base_file.files};manifest=json.loads((source/'manifest.json').read_text());cache=json.loads((Path(manifest['force_cache'])/'manifest.json').read_text());arrays=[load_arrays(Path(x)) for x in cache['dataset_paths']]
 geom=cache['geometry'];names=manifest['joint_names'];bodies=['pelvis']+[e['child'] for e in geom['edges']];surface=SurfaceSampler(ROOT/'artifacts/surface_compliance/geometry',bodies,'cpu')
 # Equal-dimensional per-link force + local moment interface for all methods.
 # Local moment preserves lever-arm information without conveying unloaded contact positions.
 methods=['random3','greedy3','greedy5'];features={k:[] for k in methods};stats={k:[] for k in methods};analytic={k:[] for k in methods};started=time.time()
 def tokens(body,local,f,rq):
  # World force and root-relative moment -> root frame. Body ID is fixed channel.
  return None
 for ri,c in enumerate(manifest['records']):
  torch.manual_seed(20261114+c['eval_id']);meta,data=arrays[c['source_index']];T=c['eval_frames'];start=c['start'];q=np.array(data['joint_pos'][start:start+T+5]);rp=np.array(data['root_pos_w'][start:start+T+5]);rq=np.array(data['root_quat_w'][start:start+T+5]);R=rot(rq)
  # Full body origin orientations reconstructed by FK from three basis points.
  origin_points=[dict(body=b,offset=[0,0,0]) for b in bodies]
  origins,_=geometry_jacobians(q,rp,rq,names,origin_points,geom)
  out={k:np.zeros((T,len(bodies),6),np.float32) for k in methods};norms={k:[] for k in methods};err={k:[] for k in methods}
  # Fixed 0.8s contact windows, final 0.1s unloading; causal application lag.
  for st in range(0,T,40):
   en=min(st+40,T);body,tri,bary=surface.sample((a.pool,));local=(tri*bary[...,None]).sum(-2).numpy();body=body.numpy();points=[dict(body=bodies[b],offset=x.tolist()) for b,x in zip(body,local)]
   pos,jac=geometry_jacobians(q[st:en+1],rp[st:en+1],rq[st:en+1],names,points,geom)
   good=pos[0,:,2]>.06
   jac[0,~good]=0
   delta=np.c_[np.diff(rp[st:en+1],axis=0), (Rotation.from_matrix(R[st+1:en+1])*Rotation.from_matrix(R[st:en]).inv()).as_rotvec(),np.diff(q[st:en+1],axis=0)].astype(np.float32)
   hs=np.minimum([0,4,8],len(delta)-1)
   selected,active=greedy_contacts(torch.tensor(jac[0:1],dtype=torch.float32),torch.tensor(delta[hs][None]),5)
   selections={'greedy5':(selected[0].numpy(),active[0].numpy())}
   s3,m3=greedy_contacts(torch.tensor(jac[0:1],dtype=torch.float32),torch.tensor(delta[hs][None]),3);selections['greedy3']=(s3[0].numpy(),m3[0].numpy())
   valid=np.flatnonzero(good);selections['random3']=(np.random.default_rng(c['eval_id']+st).choice(valid,3,replace=False),np.ones(3,bool))
   for method,(ids,mask) in selections.items():
    ids=ids[mask];current=torch.zeros(1,len(ids),3)
    for t in range(en-st):
     f=current[0].numpy();world=pos[t,ids];moment=np.cross(world-origins[st+t,body[ids]],f)
     np.add.at(out[method][st+t,:,:3],body[ids],f@R[st+t]);np.add.at(out[method][st+t,:,3:],body[ids],moment@R[st+t])
     norms[method].append([float(np.linalg.norm(f,axis=-1).sum()),float(np.linalg.norm(f,axis=-1).max()),int((np.linalg.norm(f,axis=-1)>1e-5).sum())])
     raw=torch.tensor(80*(pos[t+1,ids]-pos[t,ids]),dtype=torch.float32)[None]
     desired=limit_force(raw)
     if t>=30:desired.zero_()
     # With 30N cap, 0.1s isn't always enough to unload. Continue hold/contact
     # until zero is essential; nominal forces here are low, assert before switching.
     for _ in range(4):current=slew_force(current,desired,.005)
    if en<T and current.norm()>1e-5:raise RuntimeError('Unload window insufficient; extend it, never teleport force')

  for k in methods:features[k].append(out[k].reshape(T,-1));stats[k].extend(norms[k])
  if (ri+1)%20==0:print('clips',ri+1,'seconds',round(time.time()-started),flush=True)
 # Embed old canonical layouts into exactly the same wrench channels.
 for name in ['jacobian3','jacobian9','spring3','force37']:
  out=np.zeros((len(base['state']),len(bodies),6),np.float32)
  for i,(b,local) in enumerate(manifest['canonical_points']):
   if b not in bodies:raise ValueError(b)
   out[:,bodies.index(b),:3]+=base[name][:,i]
   # Existing points with nonzero offsets need orientation; compute below per clip.
  cursor=0
  for c in manifest['records']:
   T=c['eval_frames'];meta,data=arrays[c['source_index']];st=c['start'];q=np.asarray(data['joint_pos'][st:st+T]);rp=np.asarray(data['root_pos_w'][st:st+T]);rq=np.asarray(data['root_quat_w'][st:st+T]);R=rot(rq)
   points=[dict(body=b,offset=x) for b,x in manifest['canonical_points']];pos,_=geometry_jacobians(q,rp,rq,names,points,geom);orig,_=geometry_jacobians(q,rp,rq,names,[dict(body=b,offset=[0,0,0]) for b,x in manifest['canonical_points']],geom)
   lever=np.einsum('tji,tpj->tpi',R,pos-orig)
   for i,(b,x) in enumerate(manifest['canonical_points']):out[cursor:cursor+T,bodies.index(b),3:]+=np.cross(lever[:,i],base[name][cursor:cursor+T,i])
   cursor+=T
  features[name]=out.reshape(len(out),-1)
 features.update({k:np.concatenate(v) for k,v in list(features.items()) if isinstance(v,list)})
 np.savez_compressed(a.output/'data.npz',**features,**{k:base[k] for k in ['state','delta_q','delta_root_1','delta_root_5','split','clip','step']})
 (a.output/'manifest.json').write_text(json.dumps(dict(source=str(source),pool=a.pool,bodies=bodies,methods=list(features),seconds=time.time()-started,force_stats={k:dict(mean_total=float(np.mean(v,0)[0]),max_total=float(np.max(v,0)[0]),max_point=float(np.max(v,0)[1]),mean_count=float(np.mean(v,0)[2])) for k,v in stats.items()}),indent=2))
if __name__=='__main__':main()
