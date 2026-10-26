"""Validate every interior frame of candidate transitions, with force slew limits.
Outputs executable force labels only for fully passing intervals; no simulator claim.
"""
import json,sys,csv
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from generate_wall_support import ROOT,load_usd,add_frame
sys.path.append(str(ROOT/'active_adaptation/utils'))
from locomotion_wrench import guidance_wrench,feasible_yaw_wrench,allocate_contact_wrench

out=ROOT/'artifacts/passive_locomotion';report=json.loads((out/'planar_report.json').read_text())
model,names,meshes,_=load_usd(ROOT/'active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd')
z=min(v[:,2].min() for n,v,_ in meshes if n=='left_ankle_roll_link')
feet=[add_frame(model,s+'_sole',s+'_ankle_roll_link',np.array([0.,0.,z])) for s in ('left','right')]
hands=[model.getFrameId(s+'_hand_mimic') for s in ('left','right')];data=model.createData();mass=sum(x.mass for x in model.inertias)
records=[];csv_rows=[]
for row in report['records']:
    if not row['selected']['guidance_consistent']:continue
    m=dict(np.load(out/row['path']));scale=row['selected']['time_scale'];dt=.02*scale
    q=m['q'];v=m['v']/scale;a=m['a']/scale**2;com=m['com'];hp=m['hand_positions'];feet_pos=m['foot_positions']
    foot_vel=np.gradient(feet_pos,dt,axis=0)
    yaw=np.unwrap(Rotation.from_quat(q[:,3:7]).as_euler('xyz')[:,2]);omega=np.gradient(yaw,dt);alpha=np.gradient(omega,dt)
    previous=None;forces=[];contacts=[];torques=[];wrenches=[];desireds=[];residuals=[];rates=[];reason='ok'
    for t in range(2,len(q)-2):
        pin.computeJointJacobians(model,data,q[t]);pin.updateFramePlacements(model,data)
        hj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in hands]
        fj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in feet]
        fy=[np.arctan2(data.oMf[f].rotation[1,0],data.oMf[f].rotation[0,0]) for f in feet]
        h=pin.rnea(model,data,q[t],v[t],a[t]).copy()
        mask=(feet_pos[t,:,2]<.035)&(np.linalg.norm(foot_vel[t],axis=-1)<.35)
        guide=guidance_wrench(m['momentum'][t,:3]/mass/scale,m['momentum_rate'][t,:3]/mass/scale**2,omega[t],alpha[t])
        if row['category']=='hold':guide[:]=0.
        try:desired=feasible_yaw_wrench(hp[t],com[t],guide[:3],guide[5])
        except ValueError:reason='poor_geometry';break
        r=allocate_contact_wrench(h,hj,fj,mask,hp[t],com[t],desired,model.effortLimit[6:],previous,
            exact='planar',foot_yaws=fy,force_limit=(0. if row['category']=='hold' else 60.),max_force_rate=500.,time_step=dt)
        if not r['success']:reason=r['reason'];break
        if np.linalg.norm(r['achieved'][:3]-desired[:3])>20 or np.linalg.norm(r['achieved'][3:]-desired[3:])>5:
            reason='large_balance_correction';break
        pin.computeCentroidalMomentumTimeVariation(model,data,q[t],v[t],a[t])
        ground=np.zeros(6)
        for j in range(2):
            ground[:3]+=r['contacts'][j,:3]
            ground[3:]+=np.cross(feet_pos[t,j]-com[t],r['contacts'][j,:3])+r['contacts'][j,3:]
        residuals.append(np.max(abs(data.dhg.vector-(r['achieved']+ground+np.r_[0.,0.,-mass*9.81,0.,0.,0.]))))
        if previous is not None:rates.append(np.linalg.norm(r['forces']-previous.reshape(2,3),axis=-1).max()/dt)
        previous=r['forces'].reshape(-1);forces.append(r['forces']);contacts.append(r['contacts']);torques.append(r['torques']);wrenches.append(r['achieved']);desireds.append(desired)
    ok=reason=='ok'
    result=dict(id=row['id'],category=row['category'],time_scale=scale,success=ok,reason=reason,frames_checked=len(forces),
        frames_required=len(q)-4,max_centroidal_residual=float(max(residuals,default=0.)),
        max_force_rate_N_s=float(max(rates,default=0.)),split=row['split'])
    if ok:
        force=np.array(forces);wrench=np.array(wrenches);target=np.array(desireds)
        result.update(max_hand_force_N=float(np.linalg.norm(force,axis=-1).max()),
            max_wrench_task_error=float(abs(wrench[:,[0,1,5]]-target[:,[0,1,5]]).max()),
            max_effort_ratio=float(np.max(abs(np.array(torques))/model.effortLimit[6:])))
        folder=out/'validated';folder.mkdir(exist_ok=True)
        path=folder/f'{row["id"]:04d}_{row["category"]}.npz'
        # The first/last two source frames are excluded because finite difference
        # boundary accelerations are not reliable. Time units reflect dilation.
        np.savez_compressed(path,q=q[2:-2],v=v[2:-2],a=a[2:-2],com=com[2:-2],
            hand_positions=hp[2:-2],foot_positions=feet_pos[2:-2],hand_forces=force,
            foot_wrenches=np.array(contacts),joint_torques=np.array(torques),wrench=wrench,
            guidance_wrench=target,dt=dt,joint_names=np.array(names))
        result['path']=str(path.relative_to(out))
        for k in [0,len(force)//2,len(force)-1]:
            csv_rows.append([row['id'],row['category'],k*dt,*force[k].reshape(-1),*wrench[k]])
    records.append(result);print(result,flush=True)
summary=dict(candidates=len(records),passed=sum(r['success'] for r in records),
    by_category={k:sum(r['success'] and r['category']==k for r in records) for k in ['straight','left_turn','right_turn','hold']},records=records)
(out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'force_examples.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['transition','category','time_s','left_Fx_N','left_Fy_N','left_Fz_N','right_Fx_N','right_Fy_N','right_Fz_N','net_Fx_N','net_Fy_N','net_Fz_N','net_Mx_Nm','net_My_Nm','net_Mz_Nm']);w.writerows(csv_rows)
print('VALIDATION',json.dumps(summary,indent=2),flush=True)
