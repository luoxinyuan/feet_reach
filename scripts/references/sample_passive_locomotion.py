"""AMASS/G1 transition sampling + hand-force/contact feasibility, CPU only.
Run with .venv-reference/bin/python. No AMASS labels or targets enter a policy.
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from generate_wall_support import ROOT,load_usd,add_frame
sys.path.append(str(ROOT/'active_adaptation/utils'))
from locomotion_wrench import grasp_matrix,feasible_yaw_wrench,guidance_wrench,allocate_contact_wrench


def read_arrays(source):
    meta=json.loads((source/'meta_motion.json').read_text())
    fields=json.loads((source/'_tensordict/meta.json').read_text())
    arrays={k:np.memmap(source/'_tensordict'/f'{k}.memmap',mode='r',
                dtype=v['dtype'].split('.')[-1],shape=tuple(v['shape']))
            for k,v in fields.items() if isinstance(v,dict) and 'dtype' in v}
    return meta,arrays


def sample(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError(out/'report.json')
    meta,arrays=read_arrays(args.source)
    model,names,meshes,_=load_usd(ROOT/'active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd')
    mapping=[meta['joint_names'].index(n) for n in names]
    sole_z=min(v[:,2].min() for n,v,_ in meshes if n=='left_ankle_roll_link')
    feet=[add_frame(model,s+'_sole',s+'_ankle_roll_link',np.array([0.,0.,sole_z])) for s in ('left','right')]
    hands=[model.getFrameId(s+'_hand_mimic') for s in ('left','right')]
    data=model.createData();mass=sum(i.mass for i in model.inertias)
    rng=np.random.default_rng(args.seed);counts=Counter();reject=Counter();records=[];fps_ratios=[]
    dt=1/args.fps;categories=['straight','left_turn','right_turn','hold']
    for attempt in range(args.max_attempts):
        if all(counts[k]>=args.per_category for k in categories):break
        clip=int(rng.integers(len(meta['starts'])));start,end=meta['starts'][clip],meta['ends'][clip]
        frames=int(rng.choice([20,40,60]));margin=12
        if end-start<frames+2*margin+1:continue
        first=int(rng.integers(start+margin,end-frames-margin));last=first+frames
        root=np.array(arrays['root_pos_w'][first:last+1],dtype=float)
        quat=np.array(arrays['root_quat_w'][first:last+1],dtype=float)
        if not np.isfinite(root).all() or (np.linalg.norm(quat,axis=-1)<.9).any():reject['nonfinite']+=1;continue
        rot=Rotation.from_quat(quat[:,[1,2,3,0]])
        yaw=np.unwrap(rot.as_euler('xyz')[:,2]);delta=yaw[-1]-yaw[0]
        duration=frames*dt;distance=np.linalg.norm(root[-1,:2]-root[0,:2]);speed=distance/duration
        tilt=np.arccos(np.clip(rot.as_matrix()[:,2,2],-1,1))
        if tilt.max()>.3 or not .5<root[:,2].mean()<.95 or speed>1.2:reject['not_upright_slow_locomotion']+=1;continue
        category=None
        if .08<speed and abs(delta)<np.deg2rad(5):category='straight'
        elif np.deg2rad(6)<delta<np.deg2rad(45):category='left_turn'
        elif -np.deg2rad(45)<delta<-np.deg2rad(6):category='right_turn'
        elif speed<.04 and abs(delta)<np.deg2rad(2):category='hold'
        if category is None or counts[category]>=args.per_category:continue
        q=np.zeros((frames+2*margin+1,model.nq));sl=slice(first-margin,last+margin+1)
        q[:,:3]=np.asarray(arrays['root_pos_w'][sl],float)
        rawquat=np.asarray(arrays['root_quat_w'][sl],float)[:,[1,2,3,0]]
        r=Rotation.from_quat(rawquat);base=r[0];relative=(base.inv()*r).as_rotvec()
        q[:,3:7]=(base*Rotation.from_rotvec(savgol_filter(relative,11,3,axis=0))).as_quat()
        q[:,7:]=savgol_filter(np.asarray(arrays['joint_pos'][sl],float)[:,mapping],11,3,axis=0)
        q[:,:3]=savgol_filter(q[:,:3],11,3,axis=0)
        q=q[margin:-margin]
        if np.any(q[:,7:]<model.lowerPositionLimit[7:]-.015) or np.any(q[:,7:]>model.upperPositionLimit[7:]+.015):
            reject['joint_limits']+=1;continue
        if category=='hold' and np.max(np.ptp(q[:,7:],axis=0))>.2:reject['moving_hold']+=1;continue
        # q is now float64 smoothed G1 coordinates, quaternion xyzw for Pinocchio.
        v=np.zeros((len(q),model.nv))
        for t in range(1,len(q)-1):v[t]=(pin.difference(model,q[t],q[t+1])-pin.difference(model,q[t],q[t-1]))/(2*dt)
        v[0]=v[1];v[-1]=v[-2];a=np.gradient(v,dt,axis=0,edge_order=2)
        q[:,:2]-=q[0,:2].copy()
        foot_pos=[]
        for qt in q:
            pin.framesForwardKinematics(model,data,qt);foot_pos.append([data.oMf[f].translation.copy() for f in feet])
        foot_pos=np.array(foot_pos)
        # One constant floor translation, never per-frame floor clamping.
        floor=foot_pos[:,:,2].min();q[:,2]-=floor;foot_pos[:,:,2]-=floor
        foot_vel=np.gradient(foot_pos,dt,axis=0);contact=(foot_pos[:,:,2]<.035)&(np.linalg.norm(foot_vel,axis=-1)<.35)
        if not contact.any(-1).all():reject['no_plausible_stance']+=1;continue
        com=[];momentum=[];hdot=[];hand_pos=[]
        for qt,vt,at in zip(q,v,a):
            pin.computeCentroidalMomentumTimeVariation(model,data,qt,vt,at)
            com.append(data.com[0].copy());momentum.append(data.hg.vector.copy());hdot.append(data.dhg.vector.copy())
            pin.framesForwardKinematics(model,data,qt);hand_pos.append([data.oMf[f].translation.copy() for f in hands])
        com=np.array(com);momentum=np.array(momentum);hdot=np.array(hdot);hand_pos=np.array(hand_pos)
        velocity=momentum[:,:3]/mass;acceleration=hdot[:,:3]/mass
        yaw=np.unwrap(Rotation.from_quat(q[:,3:7]).as_euler('xyz')[:,2]);omega=np.gradient(yaw,dt);alpha=np.gradient(omega,dt)
        times=np.unique(np.linspace(2,len(q)-3,5).round().astype(int))
        row=dict(id=len(records),clip_id=clip,source_start=first-start,source_end=last-start,
                 category=category,split='val' if clip%5==0 else 'train',duration_s=duration,
                 displacement_m=distance,yaw_change_deg=float(np.rad2deg(delta)),samples=[])
        forces=[];contacts=[];torques=[];guides=[];adjusted=[];achieved=[];previous=None
        for t in times:
            pin.computeJointJacobians(model,data,q[t]);pin.updateFramePlacements(model,data)
            hj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in hands]
            fj=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).copy() for f in feet]
            foot_yaws=[np.arctan2(data.oMf[f].rotation[1,0],data.oMf[f].rotation[0,0]) for f in feet]
            h=pin.rnea(model,data,q[t],v[t],a[t]).copy()
            ideal=guidance_wrench(velocity[t],acceleration[t],omega[t],alpha[t])
            if category=='hold':ideal=np.zeros(6)
            A=grasp_matrix(hand_pos[t],com[t]);unreachable=ideal-A@np.linalg.lstsq(A,ideal,rcond=None)[0]
            try:desired=feasible_yaw_wrench(hand_pos[t],com[t],ideal[:3],ideal[5])
            except ValueError:
                row['samples'].append(dict(frame=int(t),success=False,reason='poor_yaw_geometry'))
                forces.append(np.full((2,3),np.nan));contacts.append(np.full((2,6),np.nan));torques.append(np.full(29,np.nan));achieved.append(np.full(6,np.nan))
                guides.append(ideal);adjusted.append(np.full(6,np.nan))
                continue
            result=allocate_contact_wrench(h,hj,fj,contact[t],hand_pos[t],com[t],desired,model.effortLimit[6:],previous,force_limit=(0. if category=='hold' else 60.),foot_yaws=foot_yaws)
            entry=dict(frame=int(t),success=result['success'],reason=result['reason'],grasp_rank=int(np.linalg.matrix_rank(A)),
                       raw_unreachable_norm=float(np.linalg.norm(unreachable)),raw_force_N=ideal[:3].tolist(),raw_yaw_Nm=float(ideal[5]),
                       desired_wrench=desired.tolist(),contact=contact[t].tolist())
            if result['success']:
                previous=result['forces'].reshape(-1)
                # Independent world centroidal closure against Pinocchio dh/dt.
                ground=np.zeros(6)
                for j in range(2):
                    ground[:3]+=result['contacts'][j,:3]
                    ground[3:]+=np.cross(foot_pos[t,j]-com[t],result['contacts'][j,:3])+result['contacts'][j,3:]
                closure=hdot[t]-(result['achieved']+ground+np.r_[0.,0.,-mass*9.81,0.,0.,0.])
                entry.update(force_max_N=float(np.linalg.norm(result['forces'],axis=-1).max()),
                    centroidal_residual=float(np.max(abs(closure))),equilibrium_residual=result['equilibrium_residual'],
                    effort_ratio=result['effort_ratio'],wrench_error=result['wrench_error'])
                forces.append(result['forces']);contacts.append(result['contacts']);torques.append(result['torques']);achieved.append(result['achieved'])
            else:
                forces.append(np.full((2,3),np.nan));contacts.append(np.full((2,6),np.nan));torques.append(np.full(29,np.nan));achieved.append(np.full(6,np.nan))
            guides.append(ideal);adjusted.append(desired);row['samples'].append(entry)
        row['feasible']=bool(len(row['samples'])==len(times) and all(s['success'] for s in row['samples']))
        # Each transition retains the full motion, so gait phase is never replaced
        # by an impossible static mid-swing pose. Forces are collocation samples.
        path=out/'transitions'/f'{len(records):04d}_{category}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,q=q,v=v,a=a,com=com,momentum=momentum,momentum_rate=hdot,
            hand_positions=hand_pos,foot_positions=foot_pos,contact_mask=contact,collocation_frames=times,
            hand_forces=np.array(forces),foot_wrenches=np.array(contacts),joint_torques=np.array(torques),
            raw_guidance=np.array(guides),desired_wrench=np.array(adjusted),achieved_wrench=np.array(achieved),
            fps=args.fps,joint_names=np.array(names))
        row['path']=str(path.relative_to(out));records.append(row);counts[category]+=1
        source_speed=np.linalg.norm(np.asarray(arrays['root_lin_vel_w'][first:last+1,:2],float),axis=-1).mean()
        if source_speed>.1:fps_ratios.append(float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=-1).mean()/dt/source_speed))
        print('TRANSITION',row['id'],category,'feasible',row['feasible'],dict(counts),flush=True)
    samples=[s for r in records for s in r['samples']];ok=[s for s in samples if s['success']]
    report=dict(source=str(args.source),source_frames=len(arrays['root_pos_w']),source_clips=len(meta['starts']),
        assumed_fps=args.fps,fps_consistency_median=float(np.median(fps_ratios)) if fps_ratios else None,
        seed=args.seed,robot_mass_kg=mass,counts=dict(counts),rejections=dict(reject),
        transitions=len(records),feasible_transitions=sum(r['feasible'] for r in records),
        collocation_samples=len(samples),feasible_samples=len(ok),
        infeasible_reasons=dict(Counter(s['reason'] for s in samples if not s['success'])),
        max_centroidal_residual=max((s['centroidal_residual'] for s in ok),default=None),
        max_force_N=max((s['force_max_N'] for s in ok),default=None),
        max_effort_ratio=max((s['effort_ratio'] for s in ok),default=None),
        raw_six_dof_incompatible_samples=sum(s.get('raw_unreachable_norm',0)>1e-5 for s in samples),
        records=records)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REPORT',json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/xl521/hiearchical_compliance/dataset/amass'))
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/passive_locomotion')
    p.add_argument('--fps',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--per-category',type=int,default=64);p.add_argument('--max-attempts',type=int,default=30000)
    sample(p.parse_args())
