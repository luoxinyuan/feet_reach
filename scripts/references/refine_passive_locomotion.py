"""Test time dilation and joint hand/contact allocation on sampled AMASS transitions."""
import json,sys,argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from generate_wall_support import ROOT,load_usd,add_frame
sys.path.append(str(ROOT/'active_adaptation/utils'))
from locomotion_wrench import guidance_wrench,feasible_yaw_wrench,allocate_contact_wrench,grasp_matrix

parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['planar','relaxed'],default='planar');args=parser.parse_args()
out=ROOT/'artifacts/passive_locomotion';source=json.loads((out/'report.json').read_text())
model,names,meshes,_=load_usd(ROOT/'active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd')
z=min(v[:,2].min() for n,v,_ in meshes if n=='left_ankle_roll_link')
feet=[add_frame(model,s+'_sole',s+'_ankle_roll_link',np.array([0.,0.,z])) for s in ('left','right')]
hands=[model.getFrameId(s+'_hand_mimic') for s in ('left','right')];mass=sum(x.mass for x in model.inertias)
data=model.createData()
records=[];scale_stats=Counter()
for row in source['records']:
    m=dict(np.load(out/row['path']));q=m['q'];foot=m['foot_positions'];com=m['com'];hp=m['hand_positions']
    yaw=np.unwrap(Rotation.from_quat(q[:,3:7]).as_euler('xyz')[:,2]);om=np.gradient(yaw,.02);al=np.gradient(om,.02)
    footvel=np.gradient(foot,.02,axis=0)
    attempts=[];best=None
    for scale in (1.,1.5,2.,3.):
        snapshots=[];previous=None
        for t in m['collocation_frames']:
            v=m['v'][t]/scale;a=m['a'][t]/scale**2
            pin.computeJointJacobians(model,data,q[t]);pin.updateFramePlacements(model,data)
            hj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in hands]
            fj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in feet]
            foot_yaws=[np.arctan2(data.oMf[f].rotation[1,0],data.oMf[f].rotation[0,0]) for f in feet]
            h=pin.rnea(model,data,q[t],v,a).copy()
            contact=(foot[t,:,2]<.035)&(np.linalg.norm(footvel[t],axis=-1)/scale<.35)
            guide=guidance_wrench(m['momentum'][t,:3]/mass/scale,m['momentum_rate'][t,:3]/mass/scale**2,om[t]/scale,al[t]/scale**2)
            if row['category']=='hold':guide[:]=0.
            try:desired=feasible_yaw_wrench(hp[t],com[t],guide[:3],guide[5])
            except ValueError:desired=guide
            r=allocate_contact_wrench(h,hj,fj,contact,hp[t],com[t],desired,model.effortLimit[6:],previous,force_limit=(0. if row['category']=='hold' else 60.),exact=('planar' if args.mode=='planar' else False),foot_yaws=foot_yaws)
            item=dict(frame=int(t),success=r['success'],reason=r['reason'])
            if r['success']:
                previous=r['forces'].reshape(-1);pin.computeCentroidalMomentumTimeVariation(model,data,q[t],v,a)
                ground=np.zeros(6)
                for j in range(2):
                    ground[:3]+=r['contacts'][j,:3]
                    ground[3:]+=np.cross(foot[t,j]-com[t],r['contacts'][j,:3])+r['contacts'][j,3:]
                residual=data.dhg.vector-(r['achieved']+ground+np.r_[0.,0.,-mass*9.81,0.,0.,0.])
                item.update(hand_forces=r['forces'].tolist(),foot_wrenches=r['contacts'].tolist(),joint_torques=r['torques'].tolist(),
                    desired_wrench=desired.tolist(),achieved_wrench=r['achieved'].tolist(),contact=contact.tolist(),
                    max_force_N=float(np.linalg.norm(r['forces'],axis=-1).max()),
                    force_deviation_N=float(np.linalg.norm(r['achieved'][:3]-desired[:3])),
                    moment_deviation_Nm=float(np.linalg.norm(r['achieved'][3:]-desired[3:])),
                    yaw_deviation_Nm=float(abs(r['achieved'][5]-desired[5])),
                    centroidal_residual=float(np.max(abs(residual))),effort_ratio=r['effort_ratio'])
            snapshots.append(item)
        feasible=all(s['success'] for s in snapshots)
        good=feasible and all(s['force_deviation_N']<20 and s['moment_deviation_Nm']<5 for s in snapshots)
        attempt=dict(time_scale=scale,feasible=feasible,guidance_consistent=good,samples=snapshots)
        attempts.append(attempt);scale_stats[f'{scale}:feasible']+=feasible
        if best is None or (good and not best['guidance_consistent']) or (feasible and not best['feasible']):best=attempt
        if good:break
    record=dict(id=row['id'],category=row['category'],split=row['split'],path=row['path'],selected=best,attempts=attempts)
    records.append(record)
    print(row['id'],row['category'],'scale',best['time_scale'],'feasible',best['feasible'],'guidance',best['guidance_consistent'],flush=True)
report=dict(transitions=len(records),selected_feasible=sum(x['selected']['feasible'] for x in records),
    selected_guidance_consistent=sum(x['selected']['guidance_consistent'] for x in records),
    by_category={k:dict(total=sum(x['category']==k for x in records),
        feasible=sum(x['category']==k and x['selected']['feasible'] for x in records),
        guidance_consistent=sum(x['category']==k and x['selected']['guidance_consistent'] for x in records)) for k in source['counts']},
    scale_stats=dict(scale_stats),records=records)
(out/f'{args.mode}_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
