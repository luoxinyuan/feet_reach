"""Report frozen-test decoder gains, clip bootstrap and conditional ambiguity."""
import sys,json,argparse
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.force_mapping_eval.evaluate_v2 import GROUPS
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();out=a.input/'test';r=json.loads((out/'report.json').read_text());d=np.load(a.input/'data.npz');m=json.loads((a.input/'manifest.json').read_text());test=d['split']==2;train=d['split']==0
pred={k:np.stack([np.load(out/f'{k}_{s["seed"]}_pred.npz')['prediction'] for s in v]) for k,v in r.items()}
record=np.load(out/'state_only_11_pred.npz');y=record['target'];clip=record['clip'];clips=np.unique(clip);rng=np.random.default_rng(20261114);boot=rng.integers(0,len(clips),(2000,len(clips)));summary={}
for name,pr in pred.items():
 summary[name]={}
 for group,sl in GROUPS.items():
  error=((pr[:,:,sl]-y[None,:,sl])**2).sum(2).mean(0);base=((pred['state_only'][:,:,sl]-y[None,:,sl])**2).sum(2).mean(0)
  ce=np.array([error[clip==c].sum() for c in clips]);cb=np.array([base[clip==c].sum() for c in clips]);gain=1-ce[boot].sum(1)/cb[boot].sum(1)
  summary[name][group]=dict(nrmse=float(np.mean([v['groups'][group]['nrmse'] for v in r[name]])),seed_std=float(np.std([v['groups'][group]['nrmse'] for v in r[name]])),gain_vs_state=float(1-error.sum()/base.sum()),clip_bootstrap95=np.quantile(gain,[.025,.975]).tolist())
  if name in ['greedy3','greedy5']:
   for other in ['jacobian3','jacobian9']:
    ref=((pred[other][:,:,sl]-y[None,:,sl])**2).sum(2).mean(0);cr=np.array([ref[clip==c].sum() for c in clips]);ratios=1-ce[boot].sum(1)/cr[boot].sum(1)
    summary[name][group]['vs_'+other]=dict(sse_reduction=float(1-error.sum()/ref.sum()),clip_bootstrap95=np.quantile(ratios,[.025,.975]).tolist())
  for key in ['noise_0_1N','shuffled']:
   if key in r[name][0]:summary[name][group][key]=float(np.mean([v[key][group]['nrmse'] for v in r[name]]))
state=d['state'];state=(state-state[train].mean(0))/np.maximum(state[train].std(0),.01);xs=state[test]
indices=NearestNeighbors(n_neighbors=256,n_jobs=4).fit(xs).kneighbors(xs,return_distance=False);valid=[];neighbors=[]
for i,ii in enumerate(indices):
 ii=ii[clip[ii]!=clip[i]][:8]
 if len(ii)==8:valid.append(i);neighbors.append(ii)
valid=np.asarray(valid);neighbors=np.asarray(neighbors);gaps=((y[valid,None,:29]-y[neighbors,:29])**2).sum(-1);ambiguity={};force_stats={}
for name in r:
 if name=='state_only':continue
 f=d[name];scale=np.maximum(f[train].std(0),.1);f=f[test];fd=((f[valid,None]/scale-f[neighbors]/scale)**2).sum(-1);gap=gaps[np.arange(len(valid)),fd.argmin(-1)]
 ambiguity[name]=dict(relative_joint_gap_reduction=float(1-gap.mean()/gaps.mean()),coverage=len(valid)/len(xs),note='Nearest standardized contact token among 8 cross-clip state neighbors; descriptive, not injectivity proof.')
 width=len(m['bodies'])+7;ff=f.reshape(-1,5,width)[:,:,len(m['bodies'])+3:len(m['bodies'])+6];norm=np.linalg.norm(ff,axis=-1)
 force_stats[name]=dict(mean_total_N=float(norm.sum(-1).mean()),p95_total_N=float(np.quantile(norm.sum(-1),.95)),max_total_N=float(norm.sum(-1).max()),max_point_N=float(norm.max()),mean_active=float((norm>1e-5).sum(-1).mean()),max_active=int((norm>1e-5).sum(-1).max()))
result=dict(summary=summary,ambiguity=ambiguity,force_stats=force_stats,test_frames=len(y),test_clips=len(clips),seeds=[11,22,33]);(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# Frozen test results','','560 source-grouped clips; 15,876 held-out transitions from 113 clips; three decoder seeds. Lower NRMSE is better. This is force/motion information evaluation, not trained-policy performance.','','| Method | Joint Δq | Root translation 1 frame | Root rotation 1 frame | Root translation 5 frames | Root rotation 5 frames |','|---|---:|---:|---:|---:|---:|']
for name,groups in summary.items():lines.append('| '+name+' | '+' | '.join(f'{v["nrmse"]:.3f}' for v in groups.values())+' |')
lines+=['','Max3 improves joint prediction over old Jacobian3 with a positive paired clip-bootstrap interval. Max5 has lower average joint error than Jacobian3, but its paired interval includes zero: superiority is not established. Both remain behind fixed spring3 on several metrics; extra contacts are an experimental flexibility tradeoff, not a proven information gain.','','Force totals mean sum of magnitudes, not resultant force. All methods use the same padded continuous-contact input and 256/256 decoder. See summary.json for clip-bootstrap confidence intervals, force-shuffle/noise checks, ambiguity and force budgets.','','Selection was frozen in SELECTION.json before inspecting this test. The old representative pool is 3kp-filtered; nominal-state mapping results do not establish off-reference robustness or superior interactive compliance.']
(a.input/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
