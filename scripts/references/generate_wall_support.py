"""Generate one G1 reference with Crocoddyl, using the project's actual USD.

This is a static posture solve, not a transition trajectory. A one-step
configuration retraction and Crocoddyl terminal costs are solved by SolverDDP.
Contact statics are checked separately, including Crocoddyl contact dynamics.
"""
from pathlib import Path
import argparse
import csv
import json

import crocoddyl as croc
import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]


def rotation(q):
    a = np.r_[np.asarray(q.GetImaginary()), q.GetReal()]
    return pin.Quaternion(a / np.linalg.norm(a)).matrix()


def joint_pose(j, side):
    return pin.SE3(rotation(j.GetAttribute(f"physics:localRot{side}").Get()),
                   np.array(j.GetAttribute(f"physics:localPos{side}").Get()))


def load_usd(path):
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    model = pin.Model()
    bodies = {str(p.GetPath()): p for p in stage.Traverse()
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    children = {str(j.GetRelationship("physics:body1").GetTargets()[0]) for j in joints}
    roots = set(bodies) - children
    assert len(roots) == 1
    root = roots.pop()
    jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
    links = {root: (jid, pin.SE3.Identity())}
    names, limits, efforts = [], [], []
    pending = list(joints)
    while pending:
        progress = False
        for j in pending[:]:
            parent = str(j.GetRelationship("physics:body0").GetTargets()[0])
            child = str(j.GetRelationship("physics:body1").GetTargets()[0])
            if parent not in links:
                continue
            pj, placement = links[parent]
            p0, p1 = joint_pose(j, 0), joint_pose(j, 1)
            if j.IsA(UsdPhysics.RevoluteJoint):
                axis = j.GetAttribute("physics:axis").Get()
                joint = {"X": pin.JointModelRX, "Y": pin.JointModelRY,
                         "Z": pin.JointModelRZ}[axis]()
                jid = model.addJoint(pj, joint, placement * p0, j.GetName())
                links[child] = (jid, p1.inverse())
                names.append(j.GetName())
                limits.append(np.deg2rad([j.GetAttribute("physics:lowerLimit").Get(),
                                           j.GetAttribute("physics:upperLimit").Get()]))
                efforts.append(j.GetAttribute("drive:angular:physics:maxForce").Get())
            else:
                assert j.IsA(UsdPhysics.FixedJoint)
                links[child] = (pj, placement * p0 * p1.inverse())
            pending.remove(j)
            progress = True
        assert progress, "Disconnected USD joint graph"
    for path, (jid, placement) in links.items():
        p = bodies[path]
        mass = p.GetAttribute("physics:mass").Get()
        com = np.array(p.GetAttribute("physics:centerOfMass").Get())
        diag = np.array(p.GetAttribute("physics:diagonalInertia").Get())
        # Three 5 g marker bodies have unspecified automatic inertias in USD.
        if not np.isfinite(com).all():
            com, inertia = np.zeros(3), np.zeros((3, 3))
        else:
            r = rotation(p.GetAttribute("physics:principalAxes").Get())
            inertia = r @ np.diag(diag) @ r.T
        model.appendBodyToJoint(jid, pin.Inertia(mass, com, inertia), placement)
        model.addFrame(pin.Frame(p.GetName(), jid, 0, placement, pin.FrameType.BODY))
    limits = np.array(limits)
    model.lowerPositionLimit[7:] = limits[:, 0]
    model.upperPositionLimit[7:] = limits[:, 1]
    model.effortLimit[6:] = efforts
    assert model.nq == 36 and model.nv == 35 and len(names) == 29

    # Extract real visual meshes in each rigid body's coordinates.
    cache = UsdGeom.XformCache()
    meshes = []
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or "/visuals/" not in str(p.GetPath()):
            continue
        body = p
        while str(body.GetPath()) not in links:
            body = body.GetParent()
        mesh = UsdGeom.Mesh(p)
        vertices = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.array(cache.GetLocalToWorldTransform(p) *
                             cache.GetLocalToWorldTransform(body).GetInverse())
        vertices = vertices @ transform[:3, :3] + transform[3, :3]
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        triangles, start = [], 0
        for count in counts:
            face = indices[start:start + count]
            triangles.extend((face[0], face[k], face[k+1]) for k in range(1, count-1))
            start += count
        meshes.append((body.GetName(), vertices, np.array(triangles)))

    # Check conversion against authored zero-pose body transforms.
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    root_t = np.array(cache.GetLocalToWorldTransform(bodies[root])).T
    errors = []
    for path in links:
        authored = np.linalg.inv(root_t) @ np.array(cache.GetLocalToWorldTransform(bodies[path])).T
        calculated = data.oMf[model.getFrameId(bodies[path].GetName())].homogeneous
        errors.append(np.max(np.abs(authored-calculated)))
    assert max(errors) < 5e-4, max(errors)
    return model, names, meshes, float(max(errors))


class Retraction(croc.ActionModelAbstract):
    """One configuration step; u is a tangent displacement, NOT motor torque."""
    def __init__(self, state):
        super().__init__(state, state.nv)

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        data.xnext = np.r_[pin.integrate(self.state.pinocchio, x[:self.state.nq], u),
                          np.zeros(self.state.nv)]
        data.cost = 0.5e-8 * u.dot(u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        jq, ju = pin.dIntegrate(self.state.pinocchio, x[:self.state.nq], u)
        data.Fx[:] = 0
        data.Fu[:] = 0
        data.Fx[:self.nu, :self.nu] = jq
        data.Fu[:self.nu] = ju
        data.Lu = 1e-8 * u
        data.Luu = 1e-8 * np.eye(self.nu)


class Posture(croc.ActionModelAbstract):
    def __init__(self, state, costs):
        super().__init__(state, 0)
        self.costs = costs

    def createData(self):
        data = croc.ActionDataAbstract(self)
        data.pin = self.state.pinocchio.createData()
        data.collector = croc.DataCollectorMultibody(data.pin)
        data.costs = self.costs.createData(data.collector)
        data.costs.shareMemory(data)
        return data

    def calc(self, data, x, u=None):
        model = self.state.pinocchio
        q = x[:model.nq]
        pin.computeJointJacobians(model, data.pin, q)
        pin.updateFramePlacements(model, data.pin)
        pin.centerOfMass(model, data.pin, q)
        self.costs.calc(data.costs, x, np.zeros(0))
        data.cost = data.costs.cost
        data.xnext = x

    def calcDiff(self, data, x, u=None):
        pin.jacobianCenterOfMass(self.state.pinocchio, data.pin, x[:self.state.nq])
        self.costs.calcDiff(data.costs, x, np.zeros(0))
        data.Lx = data.costs.Lx
        data.Lxx = data.costs.Lxx


def add_frame(model, name, body, xyz):
    f = model.frames[model.getFrameId(body)]
    return model.addFrame(pin.Frame(name, f.parentJoint, model.getFrameId(body),
                                   f.placement * pin.SE3(np.eye(3), xyz), pin.FrameType.OP_FRAME))


def solve(model, names, meshes):
    sole_z = min(v[:, 2].min() for b, v, _ in meshes if b == "left_ankle_roll_link")
    hand_v = np.concatenate([v for b, v, _ in meshes if b == "left_wrist_yaw_link"])
    hand_point = hand_v[hand_v[:, 2].argmin()]
    left = add_frame(model, "left_sole", "left_ankle_roll_link", np.array([0., 0., sole_z]))
    right = add_frame(model, "right_sole", "right_ankle_roll_link", np.array([0., 0., sole_z]))
    hand = add_frame(model, "left_wall_contact", "left_wrist_yaw_link", hand_point)
    targets = {
        "left_sole": pin.SE3(np.eye(3), np.array([.17, .07, .22])),
        "right_sole": pin.SE3(np.eye(3), np.array([0., -.10, 0.])),
        "left_wall_contact": pin.SE3(pin.utils.rotate("z", np.pi/2) @ pin.utils.rotate("y", -np.pi/2), np.array([.028, .25, 1.24])),
    }
    q0 = pin.neutral(model)
    q0[:3] = [0., -.08, .77]
    initial = {"left_hip_pitch_joint": -1., "left_knee_joint": 1.5,
               "left_ankle_pitch_joint": -.5, "right_hip_pitch_joint": -.1,
               "right_knee_joint": .2, "right_ankle_pitch_joint": -.1,
               "left_shoulder_pitch_joint": 0., "left_shoulder_yaw_joint": 1.57, "left_elbow_joint": -1., "left_wrist_pitch_joint": 0.,
               "left_shoulder_roll_joint": .5, "right_shoulder_roll_joint": -.16,
               "right_shoulder_pitch_joint": .35, "right_elbow_joint": .87}
    for name, value in initial.items():
        q0[model.joints[model.getJointId(name)].idx_q] = value
    state = croc.StateMultibody(model)
    costs = croc.CostModelSum(state, 0)
    def add(name, residual, weight, activation=None):
        cost = croc.CostModelResidual(state, residual) if activation is None else \
            croc.CostModelResidual(state, activation, residual)
        costs.addCost(name, cost, weight)
    for name, target in targets.items():
        activation = None
        if name == "left_wall_contact":
            # Keep the palm plane against the wall but let the fingers turn in
            # that plane, so the solver need not twist the wrist to a fixed yaw.
            activation = croc.ActivationModelWeightedQuad(np.array([1., 1., 1., 1., 1., 0.]))
        add(name, croc.ResidualModelFramePlacement(state, model.getFrameId(name), target, 0), 1e9, activation)
    add("fingers_up", croc.ResidualModelFrameRotation(
        state, hand, targets["left_wall_contact"].rotation, 0), 30.)
    add("upright_torso", croc.ResidualModelFrameRotation(
        state, model.getFrameId("torso_link"), np.eye(3), 0), 1e5)
    relaxed_arm_weights = np.zeros(state.ndx)
    for name in names:
        if name.startswith("right_") and any(k in name for k in ("shoulder", "elbow", "wrist")):
            relaxed_arm_weights[model.joints[model.getJointId(name)].idx_v] = 1.
    add("relaxed_right_arm", croc.ResidualModelState(state, np.r_[q0, np.zeros(model.nv)], 0),
        3000., croc.ActivationModelWeightedQuad(relaxed_arm_weights))
    wrist_weights = np.zeros(state.ndx)
    for name in ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"):
        wrist_weights[model.joints[model.getJointId(name)].idx_v] = 1.
    add("neutral_support_wrist", croc.ResidualModelState(state, np.r_[q0, np.zeros(model.nv)], 0),
        2000., croc.ActivationModelWeightedQuad(wrist_weights))
    natural_weights = np.zeros(state.ndx)
    natural_weights[3:6] = 1.
    for name in names:
        if name.startswith("waist_") or name.startswith("right_") and any(k in name for k in ("hip", "knee", "ankle")):
            natural_weights[model.joints[model.getJointId(name)].idx_v] = 1.
    add("human_upright_stance", croc.ResidualModelState(state, np.r_[q0, np.zeros(model.nv)], 0),
        3000., croc.ActivationModelWeightedQuad(natural_weights))
    base_height_weights = np.zeros(state.ndx)
    base_height_weights[2] = 1.
    add("base_height", croc.ResidualModelState(state, np.r_[q0, np.zeros(model.nv)], 0),
        20000., croc.ActivationModelWeightedQuad(base_height_weights))
    add("com", croc.ResidualModelCoMPosition(state, np.array([.025, -.10, .80]), 0),
        1e7, croc.ActivationModelWeightedQuad(np.array([1., 1., 0.])))
    add("posture", croc.ResidualModelState(state, np.r_[q0, np.zeros(model.nv)], 0), .3,
        croc.ActivationModelWeightedQuad(np.r_[np.ones(3), np.full(3, 3000.),
                                               np.full(29, 5.), np.zeros(model.nv)]))
    lb = np.r_[np.full(6, -1e3), model.lowerPositionLimit[7:]+.01, np.full(model.nv, -1e3)]
    ub = np.r_[np.full(6, 1e3), model.upperPositionLimit[7:]-.01, np.full(model.nv, 1e3)]
    for name in ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"):
        index = model.joints[model.getJointId(name)].idx_v
        lb[index], ub[index] = -.55, .55
    add("limits", croc.ResidualModelState(state, state.zero(), 0), 1e7,
        croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(lb, ub)))
    x0 = np.r_[q0, np.zeros(model.nv)]
    terminal = Posture(state, costs)
    problem = croc.ShootingProblem(x0, [Retraction(state)], terminal)
    solver = croc.SolverDDP(problem)
    solver.th_stop = 1e-7
    success = solver.solve([x0.copy(), x0.copy()], [np.zeros(model.nv)], 2000)
    q = np.array(solver.xs[-1][:model.nq])
    print(f"Crocoddyl DDP: success={success}, iterations={solver.iter}, cost={solver.cost:.8g}", flush=True)
    assert success, "Crocoddyl did not converge"
    return q, targets, (right, left, hand), {"solver": "Crocoddyl SolverDDP",
        "converged": bool(success), "iterations": solver.iter, "cost": solver.cost,
        "formulation": "one-step static configuration optimization; no transition trajectory"}


def statics(model, q, ids):
    support, _, hand = ids
    data = model.createData()
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    jf = pin.getFrameJacobian(model, data, support, pin.LOCAL_WORLD_ALIGNED)
    jh = pin.getFrameJacobian(model, data, hand, pin.LOCAL_WORLD_ALIGNED)[:3]
    jt = np.column_stack((jf.T, jh.T))
    gravity = pin.computeGeneralizedGravity(model, data, q)
    effort = model.effortLimit[6:]
    def tau(f):
        return gravity[6:] - jt[6:] @ f
    def inequalities(f):
        fx, fy, fz, mx, my, mz, hx, hy, hz = f
        # Conservative rectangle within the visual sole, mu_ground=.7, mu_wall=.6.
        return np.r_[fz, .7*fz-np.hypot(fx, fy), -hy-5.,
                     -.6*hy-np.hypot(hx, hz),
                     .028*fz-abs(mx), my+.11*fz, .045*fz-my,
                     .018*fz-abs(mz), effort-np.abs(tau(f))]
    # COP_x = -My/Fz: constrain to [-0.045, 0.11], a conservative forward sole region.
    guess = np.array([0., 10., sum(i.mass for i in model.inertias)*9.81,
                      0., -10., 0., 0., -10., 0.])
    opt = minimize(lambda f: np.sum((tau(f)/effort)**2) + .0003*np.sum(f[6:]**2),
                   guess, method="SLSQP", constraints=[
                       {"type": "eq", "fun": lambda f: jt[:6]@f-gravity[:6]},
                       {"type": "ineq", "fun": inequalities}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    assert opt.success, opt.message
    f = opt.x
    torque = tau(f)
    balance = np.r_[np.zeros(6), torque] + jt@f-gravity
    assert np.max(np.abs(balance)) < 1e-6
    assert min(inequalities(f)) > -1e-6
    state = croc.StateMultibody(model)
    actuation = croc.ActuationModelFloatingBase(state)
    contacts = croc.ContactModelMultiple(state, actuation.nu)
    contacts.addContact("right_sole", croc.ContactModel6D(state, support, data.oMf[support],
                        pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.zeros(2)))
    contacts.addContact("left_hand", croc.ContactModel3D(state, hand, data.oMf[hand].translation,
                        pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.zeros(2)))
    dynamics = croc.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, croc.CostModelSum(state, actuation.nu), 0., True)
    dd = dynamics.createData()
    dynamics.calc(dd, np.r_[q, np.zeros(model.nv)], torque)
    acceleration = float(np.max(np.abs(dd.xout)))
    assert acceleration < 1e-6, acceleration
    return torque, {"support_foot": "right_sole", "support_hand": "left_wall_contact",
                    "foot_wrench_world_xyz_mxyz": f[:6].tolist(),
                    "hand_force_world_xyz": f[6:].tolist(),
                    "static_balance_max_abs": float(np.max(np.abs(balance))),
                    "crocoddyl_contact_acceleration_max_abs": acceleration,
                    "max_effort_ratio": float(np.max(np.abs(torque)/effort)),
                    "ground_friction_coefficient": .7, "wall_friction_coefficient": .6,
                    "foot_cop_xy_m": [-float(f[4]/f[2]), float(f[3]/f[2])]}


def world_meshes(model, q, meshes):
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    return [(name, vertices @ data.oMf[model.getFrameId(name)].rotation.T +
             data.oMf[model.getFrameId(name)].translation, faces)
            for name, vertices, faces in meshes]


def visualize(world, out, report):
    import trimesh
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    simplified = []
    for name, vertices, faces in world:
        mesh = trimesh.Trimesh(vertices, faces, process=True)
        if len(mesh.faces) > 4000:
            mesh = mesh.simplify_quadric_decimation(face_count=4000)
        color = "#20b8a6" if name.startswith("left_") else \
                "#f0a34c" if name.startswith("right_") else "#64748b"
        simplified.append((name, np.asarray(mesh.vertices), np.asarray(mesh.faces), color))
    wall = np.array([[-.45,.25,0],[.55,.25,0],[.55,.25,1.5],[-.45,.25,1.5]])
    floor = np.array([[-.45,-.45,0],[.55,-.45,0],[.55,.25,0],[-.45,.25,0]])
    fig = go.Figure()
    for name,v,f,c in simplified:
        fig.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
            color=c, name=name, hoverinfo="name", flatshading=False))
    for v,c,opacity,name in [(wall,"#94a3b8",.25,"Side wall y=0.25 m"),
                              (floor,"#cbd5e1",.35,"Ground")]:
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=[0,0],j=[1,2],k=[2,3],
                               color=c,opacity=opacity,name=name))
    fig.update_layout(title="G1 | photo-guided wall-supported knee raise · left foot lifted 22 cm",
        scene=dict(aspectmode="data",xaxis_title="X / forward (m)",yaxis_title="Y / left (m)",
                   zaxis_title="Z / up (m)",camera=dict(eye=dict(x=1.7,y=-2.,z=.8))),
        margin=dict(l=0,r=0,b=0,t=55),paper_bgcolor="#f8fafc")
    fig.write_html(str(out/"reference.html"), include_plotlyjs=True)

    fig = plt.figure(figsize=(15,8),facecolor="#f8fafc")
    for i,(azim,title) in enumerate([(-40,"Perspective"),(-90,"Side view: knee forward")]):
        ax = fig.add_subplot(1,2,i+1,projection="3d",computed_zorder=False)
        ax.set_facecolor("#f8fafc")
        ax.add_collection3d(Poly3DCollection([wall],facecolor="#a5b4c4",alpha=.18,zorder=0))
        ax.add_collection3d(Poly3DCollection([floor],facecolor="#dae2e9",alpha=.5,zorder=0))
        triangles = np.concatenate([v[f] for _,v,f,_ in simplified])
        colors = [c for _,_,f,c in simplified for _ in f]
        ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,
                                             shade=True,linewidth=0,zorder=2))
        ax.scatter([.028],[.25],[1.24],color="#e11d48",s=40,zorder=5)
        ax.plot([.17,.17],[.07,.07],[0,.22],color="#149c8d",linestyle="--",linewidth=2,zorder=5)
        ax.text(.20,.07,.09,"22 cm",color="#0c786d",fontsize=11,zorder=6)
        ax.set(xlim=(-.35,.5),ylim=(-.4,.35),zlim=(0,1.5),xlabel="X / forward (m)",ylabel="Y (m)",zlabel="Z (m)")
        ax.set_box_aspect((.85,.75,1.5)); ax.view_init(elev=10,azim=azim)
        ax.set_title(title,fontsize=13,pad=0)
    fig.suptitle("G1 · photo-guided wall-supported knee raise",fontsize=22,fontweight="bold",y=.96)
    fig.text(.5,.89,"LEFT: wall contact + lifted foot     |     RIGHT: support foot     |     29 joints · Crocoddyl DDP",
             ha="center",fontsize=12,color="#475569")
    fig.text(.5,.035,f"Static balance verified  •  Wall normal force {-report['statics']['hand_force_world_xyz'][1]:.1f} N"
             f"  •  Peak joint effort {100*report['statics']['max_effort_ratio']:.1f}% of limit",
             ha="center",fontsize=12,color="#334155")
    fig.savefig(out/"reference.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd",type=Path,default=ROOT/"active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/wall_support")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    model,names,meshes,conversion_error = load_usd(args.usd)
    q,targets,ids,solver = solve(model,names,meshes)
    torque,force_report = statics(model,q,ids)
    data = model.createData()
    pin.framesForwardKinematics(model,data,q)
    errors = {}
    for name,target in targets.items():
        residual = pin.log6(target.inverse()*data.oMf[model.getFrameId(name)]).vector.copy()
        if name == "left_wall_contact":
            residual[5] = 0.  # Rotation in the wall plane is intentionally free.
        errors[name] = float(np.linalg.norm(residual))
    world = world_meshes(model,q,meshes)
    clearance = min(v[:,2].min() for n,v,_ in world if n=="left_ankle_roll_link")
    wall_gap = min(.25-v[:,1].max() for _,v,_ in world)
    ground_gap = min(v[:,2].min() for _,v,_ in world)
    assert max(errors.values()) < 1e-4, errors
    print("Mesh gaps:", clearance, ground_gap, wall_gap, flush=True)
    assert clearance > .17 and ground_gap > -1e-4 and wall_gap > -1e-4
    assert np.all(q[7:] >= model.lowerPositionLimit[7:]) and np.all(q[7:] <= model.upperPositionLimit[7:])
    wrist_angles = {name: float(np.rad2deg(q[model.joints[model.getJointId(name)].idx_q]))
                    for name in ("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")}
    assert max(abs(v) for v in wrist_angles.values()) < 32., wrist_angles
    palm_normal = data.oMf[model.getFrameId("left_wall_contact")].rotation[:, 2]
    palm_normal_error = float(np.linalg.norm(palm_normal - np.array([0., -1., 0.])))
    assert palm_normal_error < 1e-4
    report = {"robot":"G1 29 DoF", "source_usd":str(args.usd.relative_to(ROOT)),
        "crocoddyl_version":croc.__version__,"pinocchio_version":pin.__version__,
        "quaternion_order":"xyzw", "joint_angle_unit":"radian", "solver":solver,
        "support_foot":"right", "lifted_foot":"left", "support_hand":"left",
        "scene_config":"cfg/objects/wall_support_reference.yaml", "wall_height_m":1.5,
        "wall_plane":{"axis":"y", "position_m":.25, "force_normal_world":[0., -1., 0.]},
        "human_reference_sources":[
            "https://www.liveup.org.au/resources/strength-exercise/hip-exercises-for-older-people",
            "https://www.hiza-seitai-mizuharu.jp/blog.kataashidachi"],
        "retargeting_method":"Manually matched pose features from photographs, adapted to G1 proportions; not metric pose reconstruction.",
        "joint_names":names,"joint_positions_rad":q[7:].tolist(),
        "joint_positions_deg":np.rad2deg(q[7:]).tolist(),"base_position_world":q[:3].tolist(),
        "base_quaternion_xyzw":q[3:7].tolist(),"q_pinocchio":q.tolist(),
        "joint_torques_nm":torque.tolist(),"statics":force_report,
        "validation":{"usd_zero_pose_transform_max_error":conversion_error,
                      "left_wrist_angles_deg":wrist_angles, "palm_normal_error":palm_normal_error,
                      "frame_pose_errors":errors,"raised_foot_mesh_clearance_m":float(clearance),
                      "minimum_mesh_ground_gap_m":float(ground_gap),"minimum_mesh_wall_gap_m":float(wall_gap)},
        "limitations":["Static reference only; no transition or feedback-policy tracking test.",
                       "No full self-collision check; ground/wall gaps checked using visual vertices.",
                       "Contact friction assumed; sole uses a conservative support rectangle.",
                       "Three USD 5 g mimic markers modeled as point masses at their origins."],
        "targets":{n:{"position":t.translation.tolist(),"rotation":t.rotation.tolist()} for n,t in targets.items()}}
    (args.output/"reference.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez(args.output/"reference.npz",joint_names=np.array(names),joint_pos=q[None,7:],
             joint_vel=np.zeros((1,29)),root_pos=q[None,:3],root_quat_xyzw=q[None,3:7],
             root_quat_wxyz=q[None,[6,3,4,5]],q_pinocchio=q,torque=torque)
    with (args.output/"joint_angles.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(["joint_name","angle_rad","angle_deg","static_torque_nm"])
        writer.writerows(zip(names,q[7:],np.rad2deg(q[7:]),torque))
    visualize(world,args.output,report)
    print(json.dumps({"validation":report["validation"],"statics":force_report},indent=2))
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
