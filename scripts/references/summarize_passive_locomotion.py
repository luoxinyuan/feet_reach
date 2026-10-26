"""Summarize the offline validation and plot a passing moving transition."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'artifacts/passive_locomotion'
r=json.loads((out/'report.json').read_text());p=json.loads((out/'planar_report.json').read_text());v=json.loads((out/'validation.json').read_text())
summary=dict(sampled_transitions=r['transitions'],strict_wrench_feasible=r['feasible_transitions'],
    planar_collocation_feasible=p['selected_feasible'],guidance_consistent=p['selected_guidance_consistent'],
    full_frame_passed=v['passed'],by_category=v['by_category'])
passing=[x for x in v['records'] if x['success'] and x['category']!='hold']
if passing:
    row=sorted(passing,key=lambda x:(x['category']!='straight',x['id']))[0]
    m=np.load(out/row['path']);t=np.arange(len(m['q']))*float(m['dt']);forces=m['hand_forces'];w=m['wrench'];g=m['guidance_wrench']
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axs[0,0].plot(m['com'][:,0],m['com'][:,1]);axs[0,0].scatter(*m['com'][0,:2],label='start');axs[0,0].set(xlabel='COM x (m)',ylabel='COM y (m)',title='Reference COM path');axs[0,0].axis('equal')
    for hand,name in enumerate(['Left','Right']):
        axs[0,1].plot(t,np.linalg.norm(forces[:,hand],axis=-1),label=name)
    axs[0,1].axhline(60,color='gray',ls=':');axs[0,1].set(xlabel='Time (s)',ylabel='Hand force norm (N)',title='Applied point forces');axs[0,1].legend()
    for i,label in enumerate(['Fx','Fy','Fz']):axs[1,0].plot(t,w[:,i],label=label)
    axs[1,0].set(xlabel='Time (s)',ylabel='Net hand force (N)');axs[1,0].legend()
    for i,label in enumerate(['Mx','My','Mz']):axs[1,1].plot(t,w[:,i+3],label=label)
    axs[1,1].plot(t,g[:,5],'k--',label='Mz requested');axs[1,1].set(xlabel='Time (s)',ylabel='Moment about COM (Nm)');axs[1,1].legend()
    fig.suptitle(f'Offline AMASS/G1 transition {row["id"]}: {row["category"]}, time scale {row["time_scale"]}')
    fig.savefig(out/'force_example.png',dpi=160);plt.close(fig)
    k=len(t)//2;summary['example']=dict(id=row['id'],category=row['category'],time_s=float(t[k]),left_force_N=forces[k,0].tolist(),right_force_N=forces[k,1].tolist(),net_wrench=w[k].tolist(),max_hand_force_N=row['max_hand_force_N'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
