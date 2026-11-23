"""Matched decoder; validation-only development, explicit frozen final test."""
import sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.force_mapping_eval.evaluate_v2 import fit,metrics,GROUPS
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['state_only','jacobian3','jacobian9','spring3','random3','greedy3','greedy5']);p.add_argument('--seeds',nargs='+',type=int,default=[11]);p.add_argument('--final',action='store_true');a=p.parse_args();torch.set_num_threads(2)
d=np.load(a.input/'data.npz');out=a.input/('test' if a.final else 'validation');out.mkdir(exist_ok=True)
if a.final and not (a.input/'SELECTION.json').exists():raise RuntimeError('Freeze selection before revealing test metrics')
split=d['split'];tr=split==0;ev=split==(2 if a.final else 1);s=d['state'];s=(s-s[tr].mean(0))/np.maximum(s[tr].std(0),.01);y=np.c_[d['delta_q'],d['delta_root_1'],d['delta_root_5']];reports={};tic=time.time()
for name in a.methods:
 if name in ('state_only','gt_delta_q'):
  f=np.zeros_like(d['greedy3'])
  if name=='gt_delta_q':
   # Keep the same input width as the force decoders.
   delta_q=d['delta_q']
   f[:,:delta_q.shape[1]]=delta_q
 else:
  f=d[name]
 # Force-channel floor 0.1N, moment-channel floor 0.01Nm.
 meta=json.loads((a.input/'manifest.json').read_text())
 if name=='gt_delta_q':
  floor=.01  # rad, matching the joint-state normalization floor
 else:
  floor=np.tile([.1]*len(meta['bodies'])+[.01]*3+[.1]*4,5) if meta.get('interface')=='contact_tokens' else np.tile([.1,.1,.1,.01,.01,.01],f.shape[1]//6)
 fs=np.maximum(f[tr].std(0),floor);x=np.c_[s,(f-f[tr].mean(0))/fs].astype(np.float32)
 reports[name]=[]
 for seed in a.seeds:
  _,predict,epoch=fit(x,y,split,seed,out/f'{name}_{seed}.pt');pred=predict(x[ev]);r=dict(seed=seed,best_epoch=epoch,groups={k:metrics(y[ev,sl],pred[:,sl]) for k,sl in GROUPS.items()})
  if a.final:
   rng=np.random.default_rng(seed);sh=x[ev].copy();sh[:,s.shape[1]:]=sh[rng.permutation(len(sh)),s.shape[1]:];pr=predict(sh);r['shuffled']={k:metrics(y[ev,sl],pr[:,sl]) for k,sl in GROUPS.items()}
  if a.final and name not in ('state_only','gt_delta_q') and meta.get('interface')=='contact_tokens':
   noisy=x[ev].copy();width=len(meta['bodies'])+7;raw=f[ev].reshape(-1,5,width);active=raw[:,:,-1]>.5
   columns=np.array([[i*width+len(meta['bodies'])+3+j for j in range(3)] for i in range(5)])
   noise=rng.normal(0,.1,(len(raw),5,3))*active[...,None]
   noisy[:,s.shape[1]+columns.ravel()]+=noise.reshape(len(raw),-1)/fs[columns.ravel()]
   pr=predict(noisy);r['noise_0_1N']={k:metrics(y[ev,sl],pr[:,sl]) for k,sl in GROUPS.items()}
  reports[name].append(r)
  np.savez_compressed(out/f'{name}_{seed}_pred.npz',prediction=pred,target=y[ev],clip=d['clip'][ev])
  print(name,seed,{k:round(v['nrmse'],4) for k,v in r['groups'].items()},'seconds',round(time.time()-tic),flush=True)
 (out/'report.json').write_text(json.dumps(reports,indent=2))
