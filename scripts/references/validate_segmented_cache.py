"""Audit every stored frame and every layout switch before long training."""
import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('cache');args=p.parse_args();root=Path(args.cache)
m=json.loads((root/'manifest.json').read_text());maxforce=maxbudget=maxslew=0.;switches=0
for r in m['reports']:
    z=np.load(root/'clips'/f'{r["clip"]:05d}.npz');slots=z['slots'];f=z['forces'];t=len(f)
    assert np.isfinite(f).all() and np.all(f[slots<0]==0)
    dense=np.zeros((t,len(m['points']),3),dtype=np.float32)
    for k in range(3):
        mask=slots[:,k]>=0;dense[np.where(mask)[0],slots[mask,k]]=f[mask,k]
    assert ((np.linalg.norm(dense,axis=-1)>1e-5).sum(-1)<=3).all()
    maxforce=max(maxforce,float(np.linalg.norm(dense,axis=-1).max()))
    maxbudget=max(maxbudget,float(np.linalg.norm(dense,axis=-1).sum(-1).max()))
    maxslew=max(maxslew,float(np.linalg.norm(np.diff(dense,axis=0),axis=-1).max(initial=0)/.02))
    changed=np.where(np.any(slots[1:]!=slots[:-1],axis=1))[0];switches+=len(changed)
    assert not dense[changed].any(),r['clip']
assert maxforce<=30.001 and maxbudget<=60.001 and maxslew<=150.001
result=dict(frames=m['frames'],clips=m['clips'],accepted_frames=m['accepted_frames'],
    coverage=m['accepted_frames']/m['frames'],max_force_N=maxforce,max_budget_N=maxbudget,
    max_slew_N_s=maxslew,switches=switches,max_active_points=3,zero_before_every_switch=True,
    no_nonfinite=True,all_frames_checked=True)
(root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
