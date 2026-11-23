"""Data integrity and independent geometric reconstruction checks."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.compliance_eval.protocol import load_arrays
from active_adaptation.utils.segmented_guidance import geometry_jacobians
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());d=np.load(p/'data.npz');cache=json.loads((Path(m['force_cache'])/'manifest.json').read_text());data=[load_arrays(Path(x))[1] for x in cache['dataset_paths']]
assert all(np.isfinite(d[k]).all() for k in d.files)
groups=[{r['group'] for r in m['records'] if r['split']==i} for i in range(3)]
assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
for name in m['layouts']:
 f=d[name];assert np.linalg.norm(f,axis=-1).sum(-1).max()<60.001;assert np.linalg.norm(f,axis=-1).max()<30.001
for name in ['jacobian3','jacobian9']:
 assert not d[name][d['step']==0].any()
# Finite-difference check on joint columns, without using J to generate motion.
r=m['records'][0];a=data[r['source_index']];st=r['start'];q=np.asarray(a['joint_pos'][st:st+1],float);rp=np.asarray(a['root_pos_w'][st:st+1],float);rq=np.asarray(a['root_quat_w'][st:st+1],float)
points=cache['points'];pos,j=geometry_jacobians(q,rp,rq,m['joint_names'],points,cache['geometry']);worst=0.
for k in range(len(m['joint_names'])):
 pert=q.copy();pert[:,k]+=1e-6;pp,_=geometry_jacobians(pert,rp,rq,m['joint_names'],points,cache['geometry']);worst=max(worst,float(abs((pp-pos)/1e-6-j[:,:,:,6+k]).max()))
assert worst<1e-5,worst
# Model-informed 37-point decoder is a capacity sanity check, not main ranking.
forces=np.load(Path(m['force_cache'])/'forces.npy',mmap_mode='r');sse=signal=0.
for r in m['records']:
 if r['split']!=2:continue
 a=data[r['source_index']];st=r['start'];T=r['eval_frames'];q=np.asarray(a['joint_pos'][st:st+T+1],float);rp=np.asarray(a['root_pos_w'][st:st+T],float);rq=np.asarray(a['root_quat_w'][st:st+T],float)
 _,j=geometry_jacobians(q[:-1],rp,rq,m['joint_names'],points,cache['geometry']);f=forces[r['force_offset']:r['force_offset']+T]
 pred=np.einsum('tpdj,tpd->tj',j[:,:,:,6:],f)/m['force_K'];y=np.diff(q,axis=0);sse+=float(((pred-y)**2).sum());signal+=float((y*y).sum())
result=dict(finite=True,split_group_disjoint=True,force_caps=True,reset_force_zero=True,jacobian_finite_difference_max=worst,force37_analytic_nrmse=float(np.sqrt(sse/signal)))
(p/'validation.json').write_text(json.dumps(result,indent=2));print(result)
