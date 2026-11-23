"""G1 double-support bend and crouch references with inverse-dynamics checks."""
import argparse
import json
from pathlib import Path

import crocoddyl as croc
import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog

from generate_wall_support import ROOT, Posture, Retraction, add_frame, load_usd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', choices=('forward', 'backward', 'left', 'right', 'crouch',
                                            'yaw_left', 'yaw_right'),
                        default='forward')
    parser.add_argument('--reference-only', action='store_true',
                        help='Check the reference without the forward force-pair diagnostic')
    parser.add_argument('--angle', type=float, default=45.)
    parser.add_argument('--depth', type=float, default=.20, help='Crouch depth in meters')
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 0 < args.angle < 90 or not 0 < args.depth < .45 or args.seconds < .1:
        parser.error('requires 0 < angle < 90, 0 < depth < .45 and duration >= .1 s')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, names, meshes, fk_error = load_usd(
        ROOT / 'active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd')
    state = croc.StateMultibody(model)
    zeros = np.zeros(model.nv)
    sole_z = min(v[:, 2].min() for b, v, _ in meshes if b == 'left_ankle_roll_link')
    feet = [add_frame(model, side+'_sole', side+'_ankle_roll_link', np.array([.035, 0., sole_z]))
            for side in ('left', 'right')]
    upper = add_frame(model, 'upper_back', 'torso_link', np.array([-.05, 0., .31]))
    lower = add_frame(model, 'pelvis_back', 'pelvis', np.array([-.05, 0., -.04]))
    torso = model.getFrameId('torso_link')
    waist = model.joints[model.getJointId('waist_pitch_joint')]
    cfg = ROOT.parent / 'supermimic_force_transition/pretrained/e11c_0220000/deploy_config.json'
    deploy = json.loads(cfg.read_text())
    q0 = pin.neutral(model)
    for name, value in zip(deploy['joint_names'], deploy['default_joint_pos']):
        q0[model.joints[model.getJointId(name)].idx_q] = value
    data = model.createData()
    pin.framesForwardKinematics(model, data, q0)
    q0[2] -= np.mean([data.oMf[f].translation[2] for f in feet])
    pin.framesForwardKinematics(model, data, q0)
    targets = [pin.SE3(np.eye(3), np.r_[data.oMf[f].translation[:2], 0.]) for f in feet]
    pin.centerOfMass(model, data, q0)
    com_xy = data.com[0][:2].copy()

    axis, sign = {'forward': ('y', 1), 'backward': ('y', -1),
                  'left': ('x', -1), 'right': ('x', 1), 'crouch': ('y', 0),
                  'yaw_left': ('z', 1), 'yaw_right': ('z', -1)}[args.motion]
    pelvis = model.getFrameId('pelvis')
    forward_force_check = args.motion == 'forward' and not args.reference_only

    def solve(seed, progress):
        costs = croc.CostModelSum(state, 0)

        def add(name, residual, weight, weights=None):
            if weights is None:
                cost = croc.CostModelResidual(state, residual)
            else:
                cost = croc.CostModelResidual(state, croc.ActivationModelWeightedQuad(weights), residual)
            costs.addCost(name, cost, weight)

        for f, target in zip(feet, targets):
            add(str(f), croc.ResidualModelFramePlacement(state, f, target, 0), 1e8)
        desired = pin.utils.rotate(axis, sign*np.deg2rad(args.angle)*progress)
        add('torso', croc.ResidualModelFrameRotation(state, torso, desired, 0), 1e7)
        if axis == 'z':
            # Target actual base rotation while both feet retain their world poses.
            add('pelvis_heading', croc.ResidualModelFrameRotation(state, pelvis, desired, 0), 1e7)
        pelvis_target = q0[:3].copy()
        if args.motion == 'crouch':
            pelvis_target[2] -= args.depth*progress
        add('pelvis_position', croc.ResidualModelFrameTranslation(
            state, pelvis, pelvis_target, 0), 1e8,
            np.array([1., 1., float(args.motion == 'crouch')]))
        posture_weights = np.r_[np.zeros(2), 100., 100., 0., 100.,
                                np.full(len(names), 10.), zeros]
        add('posture', croc.ResidualModelState(state, np.r_[q0, zeros], 0), 1., posture_weights)
        add('com', croc.ResidualModelCoMPosition(state, np.r_[com_xy, .8], 0), 3000.,
            np.array([1., 1., 0.]))
        lo = np.r_[np.full(6, -1e3), model.lowerPositionLimit[7:]+.015, np.full(model.nv, -1e3)]
        hi = np.r_[np.full(6, 1e3), model.upperPositionLimit[7:]-.015, np.full(model.nv, 1e3)]
        costs.addCost('limits', croc.CostModelResidual(state, croc.ActivationModelQuadraticBarrier(
            croc.ActivationBounds(lo, hi)), croc.ResidualModelState(state, state.zero(), 0)), 1e8)
        x = np.r_[seed, zeros]
        solver = croc.SolverDDP(croc.ShootingProblem(x, [Retraction(state)], Posture(state, costs)))
        solver.th_stop = 1e-6
        ok = solver.solve([x, x], [zeros], 250)
        return np.array(solver.xs[-1][:model.nq]), bool(ok)

    progress_samples = np.linspace(0., 1., 46)
    poses, converged = [], []
    for progress in progress_samples:
        q, ok = solve(poses[-1] if poses else q0, progress)
        poses.append(q)
        converged.append(ok)
    anchor = poses[0]
    spline = CubicSpline(progress_samples, [pin.difference(model, anchor, q) for q in poses])
    times = np.linspace(0., args.seconds, round(args.seconds*100)+1)
    phase = times / args.seconds
    progress_path = 10*phase**3-15*phase**4+6*phase**5
    q_path = np.array([pin.integrate(model, anchor, spline(p)) for p in progress_path])
    dt = times[1] - times[0]
    v_path = np.zeros((len(times), model.nv))
    for i in range(1, len(times)-1):
        v_path[i] = (pin.difference(model, q_path[i], q_path[i+1]) -
                     pin.difference(model, q_path[i], q_path[i-1]))/(2*dt)
    a_path = np.gradient(v_path, dt, axis=0)
    a_path[[0, -1]] = 0.

    # Check double-support inverse dynamics with a conservative rectangular
    # foot support patch and a linearized ground friction constraint.
    inequalities = []
    for offset in (0, 6):
        for component, limit in [(0, .45), (1, .45), (3, .025), (4, .065), (5, .012)]:
            for sign in (-1., 1.):
                row = np.zeros(12)
                row[offset+component] = sign
                row[offset+2] = -limit
                inequalities.append(row)
    def check_ground(jfeet, required):
        limits = model.effortLimit[6:]
        joint_map = jfeet.T[6:]
        # Find any admissible ground wrench and all implied motor torques.
        return linprog(np.zeros(12), A_ub=np.vstack((inequalities, joint_map, -joint_map)),
                       b_ub=np.r_[np.zeros(len(inequalities)), limits+required[6:],
                                  limits-required[6:]], A_eq=jfeet.T[:6], b_eq=required[:6],
                       bounds=[(None, None), (None, None), (0, None), *[(None, None)]*3]*2,
                       method='highs')

    records = []
    prompt_levels = (0, 5, 10, 20)
    for i, (q, v, a) in enumerate(zip(q_path, v_path, a_path)):
        required = pin.rnea(model, data, q, v, a).copy()
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jfeet = np.vstack([pin.getFrameJacobian(model, data, f, pin.LOCAL_WORLD_ALIGNED) for f in feet])
        r = data.oMf[torso].rotation
        if not forward_force_check:
            base_rotation = data.oMf[pelvis].rotation
            records.append(dict(t_s=float(times[i]),
                                torso_roll_deg=float(np.rad2deg(np.arctan2(r[2,1],r[2,2]))),
                                torso_pitch_deg=float(np.rad2deg(np.arcsin(np.clip(-r[2,0],-1,1)))),
                                pelvis_yaw_deg=float(np.rad2deg(np.arctan2(base_rotation[1,0],base_rotation[0,0]))),
                                pelvis_z_m=float(q[2]),
                                foot_position_error_m=max(float(np.linalg.norm(data.oMf[f].translation-p.translation)) for f,p in zip(feet,targets)),
                                foot_angle_error_deg=max(float(np.rad2deg(np.linalg.norm(pin.log3(p.rotation.T@data.oMf[f].rotation)))) for f,p in zip(feet,targets)),
                                ground_and_motor_feasible=bool(check_ground(jfeet,required).success)))
            continue
        jpair = (pin.getFrameJacobian(model, data, upper, pin.LOCAL_WORLD_ALIGNED)[:3] -
                 pin.getFrameJacobian(model, data, lower, pin.LOCAL_WORLD_ALIGNED)[:3])
        arm = data.oMf[upper].translation - data.oMf[lower].translation
        direction = np.array([arm[2], 0., -arm[0]])
        direction /= np.linalg.norm(direction)
        g = jpair.T @ direction
        # Signed force needed if the waist-pitch motor contributes zero torque.
        signed = required[waist.idx_v] / g[waist.idx_v]
        force = float(np.clip(signed, -30., 30.))
        result = check_ground(jfeet, required - g*force)
        prompt_feasible = {str(level): bool(check_ground(jfeet, required - g*level).success)
                           for level in prompt_levels}
        r = data.oMf[torso].rotation
        records.append(dict(t_s=float(times[i]), torso_pitch_deg=float(np.rad2deg(np.arctan2(-r[2,0], r[0,0]))),
                            pelvis_xy_m=q[:2].tolist(), pelvis_z_m=float(q[2]),
                            waist_pitch_deg=float(np.rad2deg(q[waist.idx_q])),
                            foot_position_error_m=max(float(np.linalg.norm(data.oMf[f].translation-p.translation)) for f,p in zip(feet,targets)),
                            foot_angle_error_deg=max(float(np.rad2deg(np.linalg.norm(pin.log3(p.rotation.T@data.oMf[f].rotation)))) for f,p in zip(feet,targets)),
                            upper_point_world_m=data.oMf[upper].translation.tolist(),
                            lower_point_world_m=data.oMf[lower].translation.tolist(),
                            pair_forward_direction=direction.tolist(), lever_m=float(np.linalg.norm(arm[[0,2]])),
                            waist_torque_per_N=float(g[waist.idx_v]),
                            forward_prompt_feasible_by_N=prompt_feasible,
                            force_if_waist_unpowered_N=float(signed),
                            clipped_force_N=force,
                            waist_motor_torque_with_clipped_force_Nm=float(required[waist.idx_v]-g[waist.idx_v]*force),
                            ground_and_motor_feasible=bool(result.success)))
    joint_margin = np.min(np.minimum(q_path[:,7:]-model.lowerPositionLimit[7:],
                                     model.upperPositionLimit[7:]-q_path[:,7:]))
    summary = dict(motion=args.motion, target_deg=args.angle if args.motion!='crouch' else None,
                   target_depth_m=args.depth if args.motion=='crouch' else None, duration_s=args.seconds,
                   ddp_converged=sum(converged), ddp_total=len(converged), fk_import_error=fk_error,
                   max_pelvis_xy_drift_m=float(np.linalg.norm(q_path[:, :2]-anchor[:2], axis=1).max()),
                   max_foot_position_error_m=max(r['foot_position_error_m'] for r in records),
                   max_foot_angle_error_deg=max(r['foot_angle_error_deg'] for r in records),
                   min_joint_limit_margin_rad=float(joint_margin),
                   feasible_dynamics_samples=sum(r['ground_and_motor_feasible'] for r in records),
                   dynamics_samples=len(records),
                   selected_frames=[records[i] for i in np.linspace(0,len(records)-1,5,dtype=int)])
    if forward_force_check:
        summary.update(forward_prompt_feasible_samples_by_N={
                       str(level): sum(r['forward_prompt_feasible_by_N'][str(level)] for r in records)
                       for level in prompt_levels},
                   signed_force_range_N=[min(r['force_if_waist_unpowered_N'] for r in records),
                                         max(r['force_if_waist_unpowered_N'] for r in records)])
    (output/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
    trace_name = 'force_trace.json' if forward_force_check else 'reference_trace.json'
    (output/trace_name).write_text(json.dumps(records, indent=2)+'\n')
    np.savez_compressed(output/'trajectory.npz', time_s=times, q_pinocchio=q_path,
                        v_pinocchio=v_path, joint_names=names)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
