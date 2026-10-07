"""Sample a fixed-contact G1 left-foot workspace and validated smooth motions.

Run in .venv-reference. Then pack the NPZ clips with pack_foot_reach_dataset.py
in the project's training Python environment. No AMASS ground normalization.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import time

import coal
import crocoddyl as croc
import numpy as np
import pinocchio as pin
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.stats import qmc

from generate_wall_support import ROOT, load_usd, add_frame, Retraction, Posture


def body_names_from_loader():
    tree = ast.parse((ROOT / "active_adaptation/utils/motion.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "body_names_keep" for t in node.targets):
            return [n for n in ast.literal_eval(node.value) if n != "world"]
    raise RuntimeError("Cannot find training body order")


def quintic_segment(a, b, seconds, fps):
    """Exact C2 joint-space segment, including analytic velocity/acceleration."""
    count = int(np.ceil(seconds * fps))
    duration = count / fps
    t = np.arange(count + 1) / count
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = (30*t**2 - 60*t**3 + 30*t**4) / duration
    dds = (60*t - 180*t**2 + 120*t**3) / duration**2
    delta = b-a
    return a+s[:, None]*delta, ds[:, None]*delta, dds[:, None]*delta


class Workspace:
    def __init__(self, reference):
        self.model, self.names, meshes, _ = load_usd(ROOT / reference["source_usd"])
        self.q0 = np.array(reference["q_pinocchio"])
        self.sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
        self.foot = add_frame(self.model, "left_sole", "left_ankle_roll_link", np.array([0., 0., self.sole_z]))
        self.support = add_frame(self.model, "right_sole", "right_ankle_roll_link", np.array([0., 0., self.sole_z]))
        hand_vertices = np.concatenate([v for n, v, _ in meshes if n == "left_wrist_yaw_link"])
        self.hand = add_frame(self.model, "left_wall_contact", "left_wrist_yaw_link", hand_vertices[hand_vertices[:, 2].argmin()])
        self.data = self.model.createData()
        pin.computeJointJacobians(self.model, self.data, self.q0)
        pin.updateFramePlacements(self.model, self.data)
        self.anchor = self.data.oMf[self.foot].translation.copy()
        self.rotation = self.data.oMf[self.foot].rotation.copy()
        self.contact_poses = [self.data.oMf[f].copy() for f in (self.support, self.hand)]
        self.wall_y = float(reference["wall_plane"]["position_m"])
        self.root_R = pin.Quaternion(self.q0[3:7]).matrix()
        self.jt = np.column_stack([
            pin.getFrameJacobian(self.model, self.data, self.support, pin.LOCAL_WORLD_ALIGNED).T,
            pin.getFrameJacobian(self.model, self.data, self.hand, pin.LOCAL_WORLD_ALIGNED)[:3].T])
        self.leg_ids = [self.model.getJointId(n) for n in self.names[:6]]
        self.qids = np.array([self.model.joints[i].idx_q for i in self.leg_ids])
        self.vids = np.array([self.model.joints[i].idx_v for i in self.leg_ids])
        # Build the six-joint branch explicitly. The USD importer stores body
        # frames without URDF joint frames, which buildReducedModel assumes.
        self.reduced = pin.Model()
        reduced_ids = {}
        for old in self.leg_ids:
            parent = self.model.parents[old]
            placement = self.model.jointPlacements[old].copy()
            if parent not in reduced_ids:
                placement = self.data.oMi[parent] * placement
            new = self.reduced.addJoint(reduced_ids.get(parent, 0), self.model.joints[old],
                                        placement, self.model.names[old])
            reduced_ids[old] = new
            self.reduced.appendBodyToJoint(new, self.model.inertias[old], pin.SE3.Identity())
        f = self.model.frames[self.foot]
        self.reduced.addFrame(pin.Frame('left_sole', reduced_ids[f.parentJoint], 0, f.placement, pin.FrameType.OP_FRAME))
        self.reduced.lowerPositionLimit[:] = self.model.lowerPositionLimit[self.qids]
        self.reduced.upperPositionLimit[:] = self.model.upperPositionLimit[self.qids]
        self.reduced_data = self.reduced.createData()
        self.state = croc.StateMultibody(self.reduced)
        lo, hi = self.reduced.lowerPositionLimit, self.reduced.upperPositionLimit
        self.lo, self.hi = (lo+hi)/2 - .45*(hi-lo), (lo+hi)/2 + .45*(hi-lo)
        costs = croc.CostModelSum(self.state, 0)
        self.target_residual = croc.ResidualModelFramePlacement(self.state,
            self.reduced.getFrameId("left_sole"), pin.SE3(self.rotation, self.anchor), 0)
        costs.addCost("foot_pose", croc.CostModelResidual(self.state, self.target_residual), 1e7)
        reg = croc.ResidualModelState(self.state, np.r_[self.q0[self.qids], np.zeros(6)], 0)
        costs.addCost("branch", croc.CostModelResidual(self.state, reg), 1e-3)
        bounds = croc.ActivationBounds(np.r_[self.lo, np.full(6, -1e3)], np.r_[self.hi, np.full(6, 1e3)])
        limits = croc.CostModelResidual(self.state, croc.ActivationModelQuadraticBarrier(bounds),
                                        croc.ResidualModelState(self.state, self.state.zero(), 0))
        costs.addCost("limits", limits, 1e9)
        self.initial_x = np.r_[self.q0[self.qids], np.zeros(6)]
        problem = croc.ShootingProblem(self.initial_x, [Retraction(self.state)], Posture(self.state, costs))
        self.solver = croc.SolverDDP(problem)
        self.solver.th_stop = 1e-7

        # Convex hulls are conservative collision geometry built from real meshes.
        self.hulls, self.vertices = {}, {}
        for name in dict.fromkeys(n for n, _, _ in meshes):
            v = np.unique(np.concatenate([v for n, v, _ in meshes if n == name]), axis=0)
            hull = ConvexHull(v)
            points = coal.StdVec_Vec3s()
            faces = coal.StdVec_Triangle()
            for p in v:
                points.append(p)
            for t in hull.simplices:
                faces.append(coal.Triangle(*map(int, t)))
            self.hulls[name] = coal.Convex(points, faces)
            self.vertices[name] = v[hull.vertices]
        self.pairs = []
        distal = ["left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link"]
        fixed = ["pelvis", "torso_link", "right_hip_yaw_link", "right_knee_link",
                 "right_ankle_pitch_link", "right_ankle_roll_link", "right_wrist_yaw_link", "left_wrist_yaw_link"]
        self.pairs.extend((a,b) for a in distal for b in fixed)
        self.pairs.extend(("left_hip_yaw_link", b) for b in fixed[3:])
        self.frame_ids = {n:self.model.getFrameId(n) for n in self.hulls}
        self.distance_request = coal.DistanceRequest()
        self.bodies = body_names_from_loader()
        self.body_ids = [self.model.getFrameId(n) for n in self.bodies]
        self.stats = Counter()
        self.max_balance = 0.
        self.max_torque_ratio = 0.

    def solve(self, target, guess):
        self.target_residual.reference = pin.SE3(self.rotation, target)
        u = guess[self.qids]-self.q0[self.qids]
        success = self.solver.solve([self.initial_x, np.r_[guess[self.qids], np.zeros(6)]], [u], 80)
        q = self.q0.copy()
        q[self.qids] = self.solver.xs[-1][:6]
        pin.framesForwardKinematics(self.model, self.data, q)
        error = np.linalg.norm(pin.log6(pin.SE3(self.rotation, target).inverse()*self.data.oMf[self.foot]).vector)
        if not success or error > 3e-4 or np.any(q[self.qids] < self.lo-1e-6) or np.any(q[self.qids] > self.hi+1e-6):
            return None
        return q

    def geometry(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        transforms = {n:coal.Transform3s(self.data.oMf[f].rotation, self.data.oMf[f].translation)
                      for n,f in self.frame_ids.items()}
        for n in ("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
                  "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link"):
            pose = self.data.oMf[self.frame_ids[n]]
            v = self.vertices[n] @ pose.rotation.T + pose.translation
            if v[:,2].min() < .025 or v[:,1].max() > self.wall_y-.012:
                return "environment_collision"
        for a,b in self.pairs:
            result = coal.DistanceResult()
            distance = coal.distance(self.hulls[a], transforms[a], self.hulls[b], transforms[b], self.distance_request, result)
            if distance < .008:
                return "self_collision"
        for frame,target in zip((self.support,self.hand),self.contact_poses):
            if np.linalg.norm(pin.log6(target.inverse()*self.data.oMf[frame]).vector) > 1e-8:
                return "contact_drift"
        return None

    def dynamics(self, q, v=None, a=None):
        v = np.zeros(self.model.nv) if v is None else v
        a = np.zeros(self.model.nv) if a is None else a
        h = pin.rnea(self.model,self.data,q,v,a).copy()
        # f=[ground Fx,Fy,Fz,Mx,My,Mz,wall Fx,Fy,Fz]. Wall force points -Y.
        rows, rhs = [], []
        def row(coeff, bound=0.):
            r=np.zeros(9)
            for i,value in coeff.items(): r[i]=value
            rows.append(r);rhs.append(bound)
        # Conservative friction pyramids (inside circular cones).
        for sx in (-1,1):
            for sy in (-1,1):
                row({0:sx,1:sy,2:-.7})
                row({6:sx,8:sy,7:.6})
        for s in (-1,1):
            row({3:s,2:-.028}); row({5:s,2:-.018})
        row({4:-1,2:-.11});row({4:1,2:-.045})
        # 15% torque reserve relative to model effort limits.
        effort=.85*self.model.effortLimit[6:]
        A=np.vstack([rows,self.jt[6:],-self.jt[6:]])
        b=np.r_[rhs,h[6:]+effort,effort-h[6:]]
        objective=np.zeros(9);objective[7]=-1.
        result=linprog(objective,A_ub=A,b_ub=b,A_eq=self.jt[:6],b_eq=h[:6],
            bounds=[(None,None),(None,None),(1,None),(None,None),(None,None),(None,None),
                    (None,None),(-50,-3),(None,None)],method="highs")
        if not result.success: return None
        f=result.x; tau=h[6:]-self.jt[6:]@f
        error=np.max(np.abs(np.r_[np.zeros(6),tau]+self.jt@f-h))
        if error>1e-6: return None
        self.max_balance=max(self.max_balance,float(error))
        self.max_torque_ratio=max(self.max_torque_ratio,float(np.max(abs(tau)/self.model.effortLimit[6:])))
        return tau,f

    def valid(self, q):
        reason=self.geometry(q)
        if reason: return reason
        return None if self.dynamics(q) is not None else "contact_dynamics"

    def trajectory(self, waypoints, fps=50, seconds=2., hold=.4):
        chunks=[]
        for a,b in zip(waypoints[:-1],waypoints[1:]):
            # Bound speed to 1.2 rad/s and acceleration to 2.5 rad/s².
            delta=float(np.max(abs(b[self.qids]-a[self.qids])))
            duration=max(seconds,1.875*delta/1.2,np.sqrt(5.774*delta/2.5))
            leg,vel,acc=quintic_segment(a[self.qids],b[self.qids],duration,fps)
            q=np.tile(self.q0,(len(leg),1));q[:,self.qids]=leg
            v=np.zeros((len(q),self.model.nv));v[:,self.vids]=vel
            aa=np.zeros_like(v);aa[:,self.vids]=acc
            chunks.append((q[:-1],v[:-1],aa[:-1]))
            n=max(2,int(hold*fps))
            chunks.append((np.tile(b,(n,1)),np.zeros((n,self.model.nv)),np.zeros((n,self.model.nv))))
        return tuple(np.concatenate([c[k] for c in chunks]) for k in range(3))

    def export_clip(self, path, qs, vs, accs, fps):
        # All frames, including between endpoints, receive geometry + inverse dynamics checks.
        world_pos=[];world_rot=[];foot_targets=[];torques=[];forces=[]
        for q,v,a in zip(qs,vs,accs):
            reason=self.geometry(q)
            if reason: return reason
            dyn=self.dynamics(q,v,a)
            if dyn is None: return "trajectory_dynamics"
            torques.append(dyn[0]);forces.append(dyn[1])
            pin.framesForwardKinematics(self.model,self.data,q)
            world_pos.append([self.data.oMf[f].translation.copy() for f in self.body_ids])
            world_rot.append([self.data.oMf[f].rotation.copy() for f in self.body_ids])
            foot_targets.append(self.root_R.T@(self.data.oMf[self.foot].translation-q[:3]))
        wp=np.array(world_pos);wr=np.array(world_rot)
        local_pos=(wp-qs[:,:3,None].transpose(0,2,1))@self.root_R
        local_rot=np.einsum('ij,tbjk->tbik',self.root_R.T,wr)
        from scipy.spatial.transform import Rotation
        quat=Rotation.from_matrix(local_rot.reshape(-1,3,3)).as_quat().reshape(len(qs),len(self.bodies),4)
        np.savez_compressed(path,fps=np.array(fps),joint_names=np.array(self.names),body_names=np.array(self.bodies),
            root_pos=qs[:,:3].astype('f4'),root_rot=qs[:,3:7].astype('f4'),dof_pos=qs[:,7:].astype('f4'),
            local_body_pos=local_pos.astype('f4'),local_body_rot=quat.astype('f4'),
            target_foot_pos_b=np.array(foot_targets,dtype='f4'),joint_vel=vs[:,6:].astype('f4'),
            joint_acc=accs[:,6:].astype('f4'),torques=np.array(torques,dtype='f4'),contact_forces=np.array(forces,dtype='f4'))
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ROOT/'artifacts/wall_support/reference.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/foot_reach')
    p.add_argument('--grid',type=int,nargs=3,default=[7,5,6])
    p.add_argument('--sobol',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fps',type=int,default=50)
    p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--max-points',type=int,default=0,help='0 means keep all feasible samples')
    args=p.parse_args()
    if (args.output/'manifest.json').exists():
        raise FileExistsError('Choose a fresh output directory; do not mix old and new clips.')
    args.output.mkdir(parents=True,exist_ok=True)
    w=Workspace(json.loads(args.reference.read_text()))
    reason=w.valid(w.q0)
    if reason: raise RuntimeError(f'Anchor invalid: {reason}')
    lo=np.array([-.20,-.10,.07]);hi=np.array([.43,.19,.55])
    candidates=np.array(np.meshgrid(*[np.linspace(a,b,n) for a,b,n in zip(lo,hi,args.grid)],indexing='ij')).reshape(3,-1).T
    if args.sobol:
        sobol=qmc.Sobol(3,scramble=True,seed=args.seed).random_base2(int(np.ceil(np.log2(args.sobol))))[:args.sobol]
        candidates=np.vstack([candidates,lo+sobol*(hi-lo)])
    candidates=candidates[np.argsort(np.linalg.norm(candidates-w.anchor,axis=1))]
    accepted_q=[w.q0];accepted_p=[w.anchor];rejections=Counter();rejected=[]
    start=time.time()
    for index,target in enumerate(candidates):
        nearest=np.argmin(np.linalg.norm(np.array(accepted_p)-target,axis=1))
        q=w.solve(target,accepted_q[nearest])
        reason='ik' if q is None else w.valid(q)
        if reason: rejections[reason]+=1;rejected.append(target)
        else: accepted_q.append(q);accepted_p.append(target)
        if (index+1)%25==0: print(f'Samples {index+1}/{len(candidates)}: accepted={len(accepted_q)-1}, rejected={dict(rejections)}',flush=True)
        if args.max_points and len(accepted_q)-1>=args.max_points: break
    clips=[];endpoint_ids=[];frame_count=0
    for i,q in enumerate(accepted_q[1:],1):
        # Anchor returns create a connected, reproducible workspace, not unchecked random chords.
        for variant, speed_scale in enumerate((1.0, 1.5)):
            qs,vs,aa=w.trajectory([w.q0,q,w.q0],args.fps,args.seconds*speed_scale,
                                   hold=.4+.2*variant)
            if len(qs)>1000: rejections['clip_too_long']+=1;continue
            split='val' if i%5==0 else 'train'
            folder=args.output/'motions'/split;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'reach_{i:04d}_v{variant}.npz'
            reason=w.export_clip(path,qs,vs,aa,args.fps)
            if reason: rejections[reason]+=1;continue
            if i not in endpoint_ids: endpoint_ids.append(i)
            frame_count+=len(qs)
            clips.append({'file':str(path.relative_to(args.output)),'frames':len(qs),
                          'endpoint_id':i,'variant':variant,'split':split})
        if i%10==0: print(f'Trajectories {i}/{len(accepted_q)-1}: exported={len(clips)}, frames={frame_count}',flush=True)
    if not clips or not any(c['split']=='train' for c in clips): raise RuntimeError('No usable training trajectories')
    ap=np.array(accepted_p);aq=np.array(accepted_q)
    np.savez(args.output/'workspace.npz',points_world=ap,points_root=(ap-w.q0[:3])@w.root_R,
             configurations=aq,trajectory_endpoint_ids=np.array(endpoint_ids),rejected_points=np.array(rejected),
             sole_offset=np.array([0.,0.,w.sole_z]))
    manifest={'seed':args.seed,'fps':args.fps,'reference':str(args.reference),'joint_names':w.names,
        'body_names':w.bodies,'sole_offset':[0.,0.,w.sole_z],'clips':clips,'frames':frame_count,
        'candidate_count':index+1,'feasible_endpoints':len(accepted_q)-1,'connected_endpoints':len(endpoint_ids),
        'rejections':dict(rejections),'seconds_elapsed':time.time()-start,'max_inverse_dynamics_residual':w.max_balance,
        'max_effort_ratio':w.max_torque_ratio,'bounds_world':[lo.tolist(),hi.tolist()],
        'wall':{'position':[.05,.275,.75],'size':[1.,.05,1.5]},
        'coverage_note':'Sampled feasible connected subset for fixed root, right foot, left hand and horizontal left sole; not a proof of the entire reachable set.',
        'checks':['Crocoddyl IK','soft joint limits','visual convex-hull selected nonadjacent leg/body collision pairs (8 mm clearance)',
                  'ground/wall clearance','fixed support contacts','per-frame inverse dynamics with unilateral contacts, inner friction pyramids, COP and 15% torque reserve'],
        'trajectory':'C2 quintic joint interpolation; actual FK xyz is the command at every frame; endpoints from Cartesian samples.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plot_workspace(args.output,ap,endpoint_ids,w)
    print(json.dumps({k:manifest[k] for k in ('candidate_count','feasible_endpoints','connected_endpoints','frames','max_effort_ratio','seconds_elapsed')},indent=2))


def plot_workspace(out,points,ids,w):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    p=(points-w.q0[:3])@w.root_R
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d')
    ax.scatter(*p.T,s=8,c='#b8c4cf',label='Feasible endpoints')
    if ids: ax.scatter(*p[ids].T,s=16,c=p[ids,2],cmap='viridis',label='Validated trajectories')
    ax.scatter(*p[0],s=80,c='red',label='Anchor')
    ax.set(xlabel='Root X (m)',ylabel='Root Y (m)',zlabel='Root Z (m)',title='Left sole workspace | fixed left hand + right foot')
    ax.legend();fig.savefig(out/'workspace.png',dpi=160,bbox_inches='tight');plt.close(fig)
    f=go.Figure(go.Scatter3d(x=p[ids,0],y=p[ids,1],z=p[ids,2],mode='markers',marker=dict(size=3,color=p[ids,2],colorscale='Viridis')))
    f.update_layout(scene=dict(aspectmode='data',xaxis_title='Root X',yaxis_title='Root Y',zaxis_title='Root Z'))
    f.write_html(out/'workspace.html',include_plotlyjs=True)


if __name__=='__main__': main()
