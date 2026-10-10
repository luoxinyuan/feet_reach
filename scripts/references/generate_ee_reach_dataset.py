"""Crocoddyl arm reaching; double support, no wall-conditioned references.
Run with .venv-reference/bin/python. Each exported frame is checked for
joint limits, fixed support, mesh collisions and inverse-dynamics feasibility.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pinocchio as pin
import crocoddyl as croc
import coal
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from generate_wall_support import ROOT, load_usd, add_frame, Retraction, Posture
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def main():
    out=ROOT/'artifacts/ee_reach'
    if (out/'manifest.json').exists(): raise FileExistsError(out)
    model,names,meshes,_=load_usd(ROOT/'active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd')
    q0=pin.neutral(model)
    for name in names:
        val=0.
        if 'hip_pitch' in name: val=-.2
        if 'knee' in name: val=.4
        if 'ankle_pitch' in name: val=-.2
        if 'shoulder_pitch' in name: val=-.3 if name.startswith('left') else .3
        if 'shoulder_roll' in name: val=.2 if name.startswith('left') else -.2
        if 'elbow' in name: val=.8
        q0[model.joints[model.getJointId(name)].idx_q]=val
    sole_z=min(v[:,2].min() for n,v,_ in meshes if n=='left_ankle_roll_link')
    feet=[add_frame(model,s+'_sole',s+'_ankle_roll_link',np.array([0.,0.,sole_z])) for s in ('left','right')]
    hand=model.getFrameId('left_hand_mimic')
    data=model.createData();pin.framesForwardKinematics(model,data,q0)
    q0[2]-=data.oMf[feet[0]].translation[2]
    pin.computeJointJacobians(model,data,q0);pin.updateFramePlacements(model,data)
    poses=[data.oMf[f].copy() for f in feet]
    jt=np.column_stack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED).T for f in feet])
    anchor=data.oMf[hand].translation.copy();rotation=data.oMf[hand].rotation.copy()
    arm=[model.getJointId(n) for n in names if n.startswith('left_') and any(k in n for k in ('shoulder','elbow','wrist'))]
    qi=np.array([model.joints[j].idx_q for j in arm]); vi=np.array([model.joints[j].idx_v for j in arm])
    reduced=pin.Model(); ids={}
    for j in arm:
        parent=model.parents[j]; placement=model.jointPlacements[j].copy()
        if parent not in ids: placement=data.oMi[parent]*placement
        ids[j]=reduced.addJoint(ids.get(parent,0),model.joints[j],placement,model.names[j])
        reduced.appendBodyToJoint(ids[j],model.inertias[j],pin.SE3.Identity())
    f=model.frames[hand];rf=reduced.addFrame(pin.Frame('ee',ids[f.parentJoint],0,f.placement,pin.FrameType.OP_FRAME))
    reduced.lowerPositionLimit[:]=model.lowerPositionLimit[qi]+.03
    reduced.upperPositionLimit[:]=model.upperPositionLimit[qi]-.03
    state=croc.StateMultibody(reduced); costs=croc.CostModelSum(state,0)
    target=croc.ResidualModelFramePlacement(state,rf,pin.SE3(rotation,anchor),0)
    costs.addCost('ee',croc.CostModelResidual(state,target),1e7)
    costs.addCost('posture',croc.CostModelResidual(state,croc.ResidualModelState(state,np.r_[q0[qi],np.zeros(7)],0)),.1)
    bounds=croc.ActivationBounds(np.r_[reduced.lowerPositionLimit,np.full(7,-1e3)],np.r_[reduced.upperPositionLimit,np.full(7,1e3)])
    costs.addCost('limits',croc.CostModelResidual(state,croc.ActivationModelQuadraticBarrier(bounds),croc.ResidualModelState(state,state.zero(),0)),1e9)
    x0=np.r_[q0[qi],np.zeros(7)]
    solver=croc.SolverDDP(croc.ShootingProblem(x0,[Retraction(state)],Posture(state,costs)));solver.th_stop=1e-7
    hulls={}
    for n in set(n for n,_,_ in meshes):
        vv=np.unique(np.concatenate([v for b,v,_ in meshes if b==n]),axis=0);h=ConvexHull(vv)
        pts=coal.StdVec_Vec3s();tri=coal.StdVec_Triangle()
        for p in vv:pts.append(p)
        for t in h.simplices:tri.append(coal.Triangle(*map(int,t)))
        hulls[n]=coal.Convex(pts,tri)
    pairs=[(a,b) for a in ('left_elbow_link','left_wrist_yaw_link') for b in ('torso_link','pelvis','right_elbow_link','right_wrist_yaw_link','left_hip_yaw_link') if a in hulls and b in hulls]
    bodies=body_names_from_loader();bodyids=[model.getFrameId(n) for n in bodies]
    stats=Counter(); maximum={'balance':0.,'effort_ratio':0.,'support_drift':0.}
    def validate(q,v,a):
        if np.any(q[7:]<model.lowerPositionLimit[7:]+.005) or np.any(q[7:]>model.upperPositionLimit[7:]-.005):return None
        pin.framesForwardKinematics(model,data,q)
        drift=max(np.linalg.norm(pin.log6(p.inverse()*data.oMf[f]).vector) for f,p in zip(feet,poses))
        maximum['support_drift']=max(maximum['support_drift'],float(drift))
        if drift>1e-6:return None
        for aa,bb in pairs:
            pa,pb=data.oMf[model.getFrameId(aa)],data.oMf[model.getFrameId(bb)]
            d=coal.distance(hulls[aa],coal.Transform3s(pa.rotation,pa.translation),hulls[bb],coal.Transform3s(pb.rotation,pb.translation),coal.DistanceRequest(),coal.DistanceResult())
            if d<.01:return None
        h=pin.rnea(model,data,q,v,a).copy(); rows=[]
        for offset in (0,6):
            for sx in (-1,1):
                for sy in (-1,1):
                    r=np.zeros(12);r[offset:offset+3]=[sx,sy,-.7];rows.append(r)
            for moment,limit in ((3,.025),(4,.045),(5,.015)):
                for sign in (-1,1):
                    r=np.zeros(12);r[offset+moment]=sign;r[offset+2]=-limit;rows.append(r)
        effort=.85*model.effortLimit[6:]
        A=np.vstack([rows,jt[6:],-jt[6:]]);b=np.r_[np.zeros(len(rows)),h[6:]+effort,effort-h[6:]]
        bounds=[(None,None)]*12;bounds[2]=(1,None);bounds[8]=(1,None)
        res=linprog(np.zeros(12),A_ub=A,b_ub=b,A_eq=jt[:6],b_eq=h[:6],bounds=bounds,method='highs')
        if not res.success:return None
        tau=h[6:]-jt[6:]@res.x;err=np.max(abs(np.r_[np.zeros(6),tau]+jt@res.x-h))
        maximum['balance']=max(maximum['balance'],float(err));maximum['effort_ratio']=max(maximum['effort_ratio'],float(np.max(abs(tau)/model.effortLimit[6:])))
        return tau,res.x
    print('ANCHOR',anchor,'ROOT',q0[:3],flush=True)
    assert validate(q0,np.zeros(model.nv),np.zeros(model.nv)) is not None,'invalid anchor'
    clips=[];endpoints=[]
    # Independent endpoint samples. Walls are sampled only by the environment.
    samples=qmc.Halton(3,scramble=True,seed=42).random(96)
    for idx,s in enumerate(samples):
        xyz=np.array([.30,.12,.90])+s*np.array([.22,.18,.20])
        target.reference=pin.SE3(rotation,xyz)
        ok=solver.solve([x0.copy(),x0.copy()],[np.zeros(7)],150)
        q=q0.copy();q[qi]=solver.xs[-1][:7]
        pin.framesForwardKinematics(model,data,q)
        if not ok or np.linalg.norm(data.oMf[hand].translation-xyz)>.001:stats['ik']+=1;continue
        if validate(q,np.zeros(model.nv),np.zeros(model.nv)) is None:stats['endpoint']+=1;continue
        split='val' if idx%5==0 else 'train'
        for speed in (1.,1.7):
            segments=[]
            for qa,qb in ((q0,q),(q,q0)):
                duration=max(speed,1.875*np.max(abs(qb[qi]-qa[qi]))/1.5,np.sqrt(5.774*np.max(abs(qb[qi]-qa[qi]))/4.))
                pos,vel,acc=quintic_segment(qa[qi],qb[qi],duration,50)
                qq=np.tile(q0,(len(pos),1));qq[:,qi]=pos
                vv=np.zeros((len(pos),model.nv));vv[:,vi]=vel
                aa=np.zeros_like(vv);aa[:,vi]=acc
                segments.append((qq,vv,aa));segments.append((np.tile(qb,(35,1)),np.zeros((35,model.nv)),np.zeros((35,model.nv))))
            qs,vs,accs=[np.concatenate([seg[k] for seg in segments]) for k in range(3)]
            wp=[];wr=[];tau=[];forces=[];valid=True
            for qq,vv,aa in zip(qs,vs,accs):
                dyn=validate(qq,vv,aa)
                if dyn is None:valid=False;break
                pin.framesForwardKinematics(model,data,qq)
                wp.append([data.oMf[f].translation.copy() for f in bodyids]);wr.append([data.oMf[f].rotation.copy() for f in bodyids]);tau.append(dyn[0]);forces.append(dyn[1])
            if not valid:stats['trajectory']+=1;continue
            path=out/'motions'/split/f'ee_{idx:03d}_{speed:.1f}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            rot=np.array(wr);quat=Rotation.from_matrix(rot.reshape(-1,3,3)).as_quat().reshape(len(qs),len(bodies),4)
            np.savez_compressed(path,fps=np.array(50),joint_names=np.array(names),body_names=np.array(bodies),root_pos=qs[:,:3].astype('f4'),root_rot=qs[:,3:7].astype('f4'),dof_pos=qs[:,7:].astype('f4'),local_body_pos=(np.array(wp)-qs[:,None,:3]).astype('f4'),local_body_rot=quat.astype('f4'),joint_vel=vs[:,6:].astype('f4'),joint_acc=accs[:,6:].astype('f4'),torques=np.array(tau,dtype='f4'),contact_forces=np.array(forces,dtype='f4'))
            clips.append(dict(path=str(path.relative_to(out)),split=split,endpoint_id=idx,frames=len(qs),target=xyz.tolist()))
        endpoints.append(xyz.tolist());print(idx,len(clips),dict(stats),flush=True)
    assert len(clips)>=40 and any(c['split']=='val' for c in clips)
    manifest=dict(fps=50,joint_names=names,body_names=bodies,clips=clips,anchor_world=anchor.tolist(),q_anchor=q0.tolist(),checks=maximum,rejections=dict(stats),solver='Crocoddyl DDP arm pose solve + quintic interpolation + per-frame double-support inverse dynamics',collision_pairs=pairs,seed=42)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DATASET_OK',len(clips),sum(c['frames'] for c in clips),maximum,flush=True)
if __name__=='__main__':main()
