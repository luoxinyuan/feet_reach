"""Train-only global gain calibration; no per-frame scaling and no validation tuning."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.compliance_eval.protocol import load_arrays
from active_adaptation.utils.segmented_guidance import geometry_jacobians
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=float,default=20);a=p.parse_args()
m=json.loads(a.manifest.read_text());geometry=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text());points=m['layouts']['offsets37'];totals=[];maxpoints=[];rates=[];noise=[]
for r in m['records']:
 if r['split']!='train_diagnostic':continue
 meta,data=load_arrays(ROOT/'dataset'/r['dataset']);st=r['start'];T=r['frames'];q=np.asarray(data['joint_pos'][st:st+T+1],float)
 _,j=geometry_jacobians(q[:-1],np.asarray(data['root_pos_w'][st:st+T],float),np.asarray(data['root_quat_w'][st:st+T],float),meta['joint_names'],points,geometry)
 A=j[:,:,:,6:].transpose(0,3,1,2).reshape(T,29,-1);dq=np.diff(q,axis=0)
 f=np.einsum('tkn,tn->tk',np.linalg.pinv(A,rcond=1e-7),80*dq).reshape(T,len(points),3)
 norms=np.linalg.norm(f,axis=-1);totals.extend(norms.sum(1));maxpoints.extend(norms.max(1));rates.extend(np.linalg.norm(np.diff(f,axis=0),axis=-1).max(1)/.02)
constraints=dict(total_p99=a.budget/np.percentile(totals,99),point_p99=30/np.percentile(maxpoints,99),slew_p99=150/np.percentile(rates,99))
gain=80*min(constraints.values());result=dict(gain=gain,budget=a.budget,scale_constraints=constraints,train_frames=len(totals),reference_gain=80,
 total_p99_before=float(np.percentile(totals,99)),slew_p99_before=float(np.percentile(rates,99)),rule='K=80*min(B/p99(total),30/p99(max point),150/p99(max point slew)); training windows only')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
