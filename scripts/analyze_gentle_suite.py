"""Analyze paired gentle-controller evaluations without conflating no contact with low impact."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from active_adaptation.utils.gentle_contact import ContactEvents

MODELS=['CEER','A_teacher','A_student','B_teacher','B_student']
METRICS=['impulse_100ms_Ns','peak_force_N','initial_peak_N','max_rate_N_s','precontact_vx_m_s','hold_tangent_error_m','hold_force_N','residual_mean_m']

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
 records={m:json.loads((root/m/'summary.json').read_text()) for m in MODELS}
 assert all(len(v)==30 for v in records.values())
 ids=[x['trial'] for x in records['CEER']]
 assert all([x['trial'] for x in v]==ids for v in records.values())
 paired={}
 for m in MODELS[1:]:
  maxdiff=0.;hash_equal=0
  for a,b in zip(records['CEER'],records[m]):
   x=np.load(root/'CEER'/(a['trial']+'.npz'))['initial'];y=np.load(root/m/(b['trial']+'.npz'))['initial']
   maxdiff=max(maxdiff,float(np.abs(x-y).max()));hash_equal+=a['initial_sha256']==b['initial_sha256']
  paired[m]=dict(initial_exact_matches=hash_equal,initial_max_difference=maxdiff)
 report={'pairing':paired,'models':{},'comparisons':{}}
 for m,rows in records.items():
  extra=[];summed={}
  for r in rows:
   data=np.load(root/m/(r['trial']+'.npz'));ph=data['physics'];ctrl=data['control']
   f=ph[:,1];tc=r['contact_time_s'];peakidx=int(np.argmax(f));firstidx=int(np.flatnonzero(f>1)[0]) if np.any(f>1) else None
   e=ContactEvents(1,'cpu',.005);cost=np.zeros(3);peak_covered=False;active_count=rate_saturated=0
   for i,force in enumerate(f):
    previous_count=int(e.count[0]);active=bool(e.age[0]<e.window_steps)
    rate=max((force-f[i-1] if i else force)/.005,0.)
    dj,dp,g=e.advance(torch.tensor([force],dtype=torch.float32),torch.tensor([True]))
    active=active or int(e.count[0])>previous_count
    if i==peakidx:peak_covered=active
    if active:
     active_count+=1;rate_saturated+=rate>=500+5000*np.sqrt(.5)
    cost+=np.array([dj.item()*.02,dp.item()*.02,g.item()*.005])
   tangent=np.linalg.norm(ph[:,3:5]-ph[:,6:8],axis=1)
   h=ph[:,0]>r['duration'];good=(f>1)&(tangent<.03)&h
   consecutive=maxrun=0
   for g in good:
    consecutive=consecutive+1 if g else 0;maxrun=max(maxrun,consecutive)
   ex=dict(trial=r['trial'],events=int(e.count[0]),peak_time=float(ph[peakidx,0]),peak_in_penalized_window=peak_covered,
           peak_after_first_window=bool(tc is not None and peakidx>=firstidx+20),
           peak_cost_saturated=bool(firstidx is not None and f[firstidx:firstidx+20].max()>=40+60*np.sqrt(5/3)),
           rate_saturated_fraction=rate_saturated/max(active_count,1),counterfactual_impulse_cost=cost[0],
           counterfactual_peak_cost=cost[1],counterfactual_growth_cost=cost[2],
           residual_axis_saturation_fraction=float((np.abs(ctrl[:,5:8])>.145).mean()),
           hold_good_fraction=float(good[h].mean()) if h.any() else 0.,hold_success=bool(maxrun>=60 and r['stable']))
   extra.append(ex)
   for k,v in r['reward_returns'].items():summed.setdefault(k,[]).append(v)
  def stats(selected):
   d={}
   for k in METRICS:
    values=[r[k] for r in selected if r[k] is not None]
    d[k]=dict(n=len(values),median=float(np.median(values)),p90=float(np.percentile(values,90)),mean=float(np.mean(values))) if values else None
   return d
  report['models'][m]=dict(n=len(rows),success=sum(r['success'] for r in rows),hold_success=sum(r['hold_success'] for r in extra),
      stable=sum(r['stable'] for r in rows),contact=sum(r['contact'] for r in rows),
      all_contact=stats([r for r in rows if r['stable'] and r['contact']]),successful=stats([r for r in rows if r['success']]),
      mean_reward_returns={k:float(np.mean(v)) for k,v in summed.items()},
      event_diagnostics={k:float(np.mean([r[k] for r in extra])) for k in extra[0] if k!='trial'},events=extra)
 for a,b in [('CEER','A_teacher'),('CEER','A_student'),('CEER','B_teacher'),('CEER','B_student'),('A_teacher','A_student'),('B_teacher','B_student'),('A_student','B_student')]:
  d={}
  for k in METRICS:
   pairs=[(x[k],y[k]) for x,y in zip(records[a],records[b]) if x['stable'] and y['stable'] and x['contact'] and y['contact'] and x[k] is not None and y[k] is not None]
   delta=np.array([y-x for x,y in pairs]);d[k]=dict(n=len(pairs),b_lower=int((delta<0).sum()),median_delta=float(np.median(delta)))
  d['success_lost']=sum(x['success'] and not y['success'] for x,y in zip(records[a],records[b]))
  d['success_gained']=sum(not x['success'] and y['success'] for x,y in zip(records[a],records[b]))
  report['comparisons'][a+' -> '+b]=d
 (root/'analysis.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({m:{k:v for k,v in d.items() if k in ['success','hold_success','stable','contact','all_contact','event_diagnostics']} for m,d in report['models'].items()},indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,3,figsize=(15,4))
 for ax,key,title in zip(axs,['impulse_100ms_Ns','peak_force_N','hold_tangent_error_m'],['Initial 100 ms impulse (N s)','Whole-trial peak force (N)','Hold tangential error (m)']):
  values=[[r[key] for r in records[m] if r['stable'] and r['contact'] and r[key] is not None] for m in MODELS]
  ax.boxplot(values,labels=MODELS,showmeans=True);ax.tick_params(axis='x',rotation=25);ax.set_title(title);ax.grid(axis='y',alpha=.3)
 fig.suptitle('30 paired scenes; contact trials shown, task success reported separately')
 fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
