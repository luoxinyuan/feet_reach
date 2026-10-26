"""Compare independently launched, exactly matched external-force/zero trials."""
from pathlib import Path
import argparse,json,csv
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root;m=root/'metrics'
meta=json.loads((m/'pass_forced.json').read_text());rows=meta['rows'];f=np.load(m/'forced.npz');z=np.load(m/'zero.npz')
for key in ['initial_sha256','initial_public','initial_alpha','initial_delay']:np.testing.assert_array_equal(f[key],z[key])
pre=float(np.abs(f['root'][:101]-z['root'][:101]).max());assert pre<1e-4,pre
records=[]
for i,row in enumerate(rows):
 direction=np.array(row['vector']);r=f['root'][:,i];pt=f['point'][:,i];zr=z['root'][:,i];zp=z['point'][:,i]
 rd=r[300]-r[100];pd=pt[300]-pt[100];zrd=zr[300]-zr[100];zpd=zp[300]-zp[100]
 yaw=np.unwrap(f['yaw'][:,i]);zyaw=np.unwrap(z['yaw'][:,i]);yd=float(np.degrees(yaw[300]-yaw[100]));zyd=float(np.degrees(zyaw[300]-zyaw[100]));stable=not np.isfinite(f['first_fall_s'][i])
 cr=float((rd-zrd)@direction);cp=float((pd-zpd)@direction);success=stable and ((yd-zyd)*row['turn']>10 if row['turn'] else (cr if row['group']=='body' else cp)>.05)
 records.append(dict(row,stable=stable,first_fall_s=float(f['first_fall_s'][i]) if not stable else None,root_along_m=float(rd@direction),point_along_m=float(pd@direction),root_relative_point_along_m=float((pd-rd)@direction),root_along_minus_zero_m=cr,point_along_minus_zero_m=cp,yaw_deg=yd,yaw_minus_zero_deg=yd-zyd,root_travel_during_force_m=float(np.linalg.norm(rd[:2])),root_drift_after_release_m=float(np.linalg.norm(r[500,:2]-r[300,:2])),yaw_after_release_deg=float(np.degrees(yaw[500]-yaw[300])),success=bool(success),max_point_force_N=float(np.linalg.norm(f['force'][:,i],axis=-1).max()),max_net_force_N=float(np.linalg.norm(f['net_force'][:,i],axis=-1).max())))
keys=['root_along_m','point_along_m','root_relative_point_along_m','root_along_minus_zero_m','point_along_minus_zero_m','yaw_deg','yaw_minus_zero_deg','root_travel_during_force_m','root_drift_after_release_m','yaw_after_release_deg']
agg=[]
for name in dict.fromkeys(r['name'] for r in records):
 for force in [5.,10.,20.]:
  rs=[r for r in records if r['name']==name and r['force_N']==force]
  agg.append(dict(name=name,force_N=force,stable_fraction=float(np.mean([r['stable'] for r in rs])),directional_response_fraction=float(np.mean([r['success'] for r in rs])),**{k:float(np.mean([r[k] for r in rs])) for k in keys}))
zero={'physical_survival_fraction':float(np.mean(~np.isfinite(z['first_fall_s']))),'mean_root_displacement_10s_m':float(np.linalg.norm(z['root'][-1,:,:2]-z['root'][0,:,:2],axis=-1).mean()),'mean_abs_yaw_change_10s_deg':float(np.abs(np.degrees(np.unwrap(z['yaw'],axis=0)[-1]-z['yaw'][0])).mean())}
report={'checkpoint':meta['checkpoint'],'seed':42,'repeats':8,'preprompt_max_root_difference_m':pre,'paired_initial_public_and_actuator_verified':True,'zero_control':zero,'aggregate':agg,'records':records}
(root/'comparison.json').write_text(json.dumps(report,indent=2))
with (root/'aggregate.csv').open('w') as file:
 writer=csv.DictWriter(file,fieldnames=agg[0].keys());writer.writeheader();writer.writerows(agg)
print('ZERO',zero)
for r in agg:
 if r['force_N']==10:print(r['name'],'stable',r['stable_fraction'],'response',r['directional_response_fraction'],'root',round(r['root_along_m'],2),'EE',round(r['point_along_m'],2),'yaw',round(r['yaw_deg'],1),'release_m',round(r['root_drift_after_release_m'],2),'release_deg',round(r['yaw_after_release_deg'],1))
# Scientific plots, independent of the video renderer.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,10));colors=['tab:blue','tab:orange','tab:green','tab:red']
for ax,group in zip(axes.flat,['left','right','body','turn']):
 names=list(dict.fromkeys(r['name'] for r in rows if r['group']==group))
 for name,color in zip(names,colors):
  i=next(i for i,r in enumerate(rows) if r['name']==name and r['force_N']==10 and r['repeat']==0)
  if group=='turn':
   t=np.arange(501)*.02;y=np.degrees(np.unwrap(f['yaw'][:,i])-f['yaw'][0,i]);zy=np.degrees(np.unwrap(z['yaw'][:,i])-z['yaw'][0,i]);ax.plot(t,y,label=name,color=color);ax.plot(t,zy,'--',alpha=.5,color=color)
  else:
   path=f['root'][:,i,:2]-f['root'][100,i,:2];ax.plot(path[100:301,0],path[100:301,1],color=color,label=name);ax.plot(path[300:,0],path[300:,1],':',color=color);ax.scatter(path[300,0],path[300,1],color=color,s=20)
 if group=='turn':ax.axvspan(2,6,alpha=.1,color='gray');ax.set_xlabel('Time (s)');ax.set_ylabel('Unwrapped yaw (deg)');ax.set_title('Shoulder couples: dashed = zero-force control')
 else:ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Initial-frame X (m)');ax.set_ylabel('Initial-frame Y (m)');ax.set_title(group+' force: root paths; dotted = after release');ax.scatter(0,0,color='black',marker='x')
 ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('Jacobian policy: 10 N per point, replicate 0 (not performance selected)');fig.tight_layout();fig.savefig(root/'response_traces.png',dpi=160);plt.close(fig)
