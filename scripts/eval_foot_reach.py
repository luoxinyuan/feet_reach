"""Fixed live-root foot targets, quantitative student evaluation, video and web control."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', help='Local file, W&B/Forge run URL, or run:entity/project/run-id; defaults to latest pipeline final student')
    p.add_argument('--wandb-file', help='Exact checkpoint path in W&B run Files; defaults to final, then highest numbered checkpoint')
    p.add_argument('--checkpoint-cache', type=Path, default=ROOT / '.cache/wandb-checkpoints')
    p.add_argument('--targets', type=Path, help='JSON: [[x,y,z], ...] in live root frame, metres')
    p.add_argument('--points', type=int, default=20, help='Number of consecutive nearest-neighbor feasible targets')
    p.add_argument('--settle', type=float, default=1.)
    p.add_argument('--reach', type=float, default=2., help='Time allowed to reach the directly commanded target')
    p.add_argument('--hold', type=float, default=1., help='Final measurement window in seconds')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--random-foot-force', action='store_true',
                   help='Apply the existing randomized 0–20 N external-force schedule to the reaching left foot')
    p.add_argument('--output', type=Path)
    p.add_argument('--continue-on-instability', action='store_true', help='Record instability but continue all targets without resetting')
    p.add_argument('--video', action='store_true')
    p.add_argument('--web', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--web-seconds', type=float, default=0., help='0 keeps server running until Ctrl+C')
    args = p.parse_args()
    if args.points < 1 or min(args.settle, args.reach, args.hold) <= 0:
        p.error('Counts and durations must be positive')
    if args.checkpoint is None:
        if args.wandb_file:
            p.error('--wandb-file requires --checkpoint with a W&B run')
        if not (ROOT / 'outputs/wall-foot-reach/latest_pipeline.txt').is_file():
            p.error('No local pipeline found; specify --checkpoint with a local file or W&B run URL')
        pipeline = Path((ROOT / 'outputs/wall-foot-reach/latest_pipeline.txt').read_text().strip())
        matches = list((pipeline / 'finetune').rglob('checkpoint_final.pt'))
        if not matches:
            completed = sorted((ROOT/'outputs/wall-foot-reach').glob('*/finetune/**/checkpoint_final.pt'))
            if completed:
                matches = [completed[-1]]
                print(f'Latest pipeline still running; evaluating completed checkpoint: {matches[0]}')
        if len(matches) != 1: p.error('Specify --checkpoint: final finetune checkpoint not unique')
        args.checkpoint = matches[0]
    from scripts.foot_reach.checkpoints import resolve_checkpoint
    try:
        args.checkpoint = resolve_checkpoint(str(args.checkpoint), args.checkpoint_cache, args.wandb_file)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if args.output is None:
        args.output = ROOT / 'artifacts/foot_reach_eval' / time.strftime('%Y%m%d_%H%M%S')
    args.output.mkdir(parents=True, exist_ok=False)
    return args


def select_targets(args):
    import numpy as np
    workspace = np.load(ROOT / 'artifacts/foot_reach/workspace.npz')
    anchor = workspace['points_root'][0]
    if args.targets:
        targets = np.asarray(json.loads(args.targets.read_text()), dtype=float)
    else:
        candidates = np.unique(workspace['points_root'], axis=0)
        candidates = candidates[np.linalg.norm(candidates-anchor, axis=1) > 1e-9]
        if len(candidates) < args.points:
            raise ValueError(f'Only {len(candidates)} distinct non-anchor feasible points; requested {args.points}')
        chosen = []
        current = anchor
        for _ in range(args.points):
            index = int(np.linalg.norm(candidates-current, axis=1).argmin())
            current = candidates[index].copy()
            chosen.append(current)
            candidates = np.delete(candidates, index, axis=0)
        targets = np.asarray(chosen)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all() or not len(targets):
        raise ValueError('Targets must be a finite nonempty Nx3 array')
    return targets, anchor, workspace['points_root'].min(0), workspace['points_root'].max(0)


def write_reports(output, trials, planned_targets=None):
    """Keep failures in denominators; accuracy of completed trials is labelled."""
    points = []
    for point in sorted({r['point'] for r in trials}):
        rows = [r for r in trials if r['point'] == point]
        completed = [r for r in rows if r['completed_hold'] and r['mean_error_m'] is not None]
        points.append(dict(point=point, target_root_xyz_m=rows[0]['target_root_xyz_m'],
            trials=len(rows), successes=sum(r['success'] for r in rows),
            failures=sum(r['failed'] for r in rows),
            mean_error_completed_trials_m=(sum(r['mean_error_m'] for r in completed)/len(completed)
                                           if completed else None)))
    planned_targets = len(trials) if planned_targets is None else planned_targets
    summary = dict(trials=trials, per_point=points,
        planned_targets=planned_targets, attempted_targets=len(trials),
        unattempted_targets=planned_targets-len(trials),
        all_targets_evaluated=len(trials)==planned_targets and all(r['completed_hold'] for r in trials),
        sequence_completed=len(trials)==planned_targets and all(r['success'] for r in trials),
        success_rate=sum(r['success'] for r in trials)/len(trials),
        failure_rate=sum(r['failed'] for r in trials)/len(trials),
        note='Hold frames only. Rates use attempted targets only; unattempted targets are reported separately. Drift is relative to the initial reset. Success means the entire trial stayed stable; accuracy does not determine success. Per-point mean uses completed holds only.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader(); writer.writerows(trials)


def plot_contact_forces(output):
    """Plot filtered wall-hand and ground-foot force magnitudes over sim time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (output/'samples.csv').open(newline='') as f:
        samples = list(csv.DictReader(f))
    if not samples:
        return
    times = [float(row['time_s']) for row in samples]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    traces = [
        ('left_hand_wall_force_norm_N', 'Left hand ↔ wall', '#2563eb'),
        ('right_support_foot_force_norm_N', 'Right support foot contact force', '#ea580c'),
    ]
    for ax, (field, title, color) in zip(axes, traces):
        values = [float(row[field]) for row in samples]
        ax.plot(times, values, color=color, linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel('Force magnitude (N)')
        ax.grid(True, alpha=.25)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle('Filtered contact forces during foot-reach evaluation')
    fig.savefig(output/'contact_forces.png', dpi=160)
    plt.close(fig)


def main():
    args = arguments()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    targets, anchor, lower, upper = select_targets(args)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'vecnorm' not in state or 'cfg' not in state:
        raise ValueError('Checkpoint must include cfg and vecnorm; refusing unnormalised evaluation')
    if state['policy'].get('last_phase') not in ('adapt', 'finetune'):
        raise ValueError('Use a trained student checkpoint from adapt or finetune')
    cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
    OmegaConf.set_struct(cfg, False)
    # Older checkpoints predate the optional geometry encoder setting.
    # Their policy weights contain no geometry modules, so zero points is
    # the compatible configuration default.
    if 'geometry_points' not in cfg.algo:
        cfg.algo.geometry_points = 0
    if not cfg.task.command._target_.endswith('.WallFootReachCommand'):
        raise ValueError('Checkpoint is not a foot reach policy')
    has_wall = any(obj.get('type') == 'wall' for obj in (cfg.task.objects or []))
    cfg.checkpoint_path = str(args.checkpoint)
    cfg.vecnorm = 'eval'; cfg.seed = args.seed; cfg.headless = True
    cfg.task.num_envs = 1; cfg.task.max_episode_length = 100000000
    cfg.task.eval_contact_forces = True
    cfg.task.command.student_train = True
    if 'foot_force' not in cfg.task.command:
        cfg.task.command.foot_force = OmegaConf.create({})
    cfg.task.command.foot_force.enabled = args.random_foot_force
    cfg.task.command.foot_force.train_only = False
    cfg.task.command.foot_force.force_range = [0., 20.]
    cfg.task.command.disable_motion_finish = True
    cfg.task.command.dataset.fix_ds = 0; cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.termination = {}  # Evaluate failures explicitly, never auto-reset them away.
    cfg.task.viewer.resolution = [960, 720]
    cfg.task.viewer.eye = [2.1, -2.4, 1.5]; cfg.task.viewer.lookat = [0., 0., .75]
    cfg.eval_render = args.video or args.web
    cfg.app = dict(headless=True, enable_cameras=cfg.eval_render)
    cfg.wandb.mode = 'disabled'
    (args.output / 'targets.json').write_text(json.dumps(targets.tolist(), indent=2))
    (args.output / 'config.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        coordinate_frame='live root, full quaternion; left sole centre; metres', seed=args.seed,
        external_force_enabled=args.random_foot_force,
        external_force=dict(
            body='left_ankle_roll_link at left sole centre',
            range_N=[0., 20.],
            direction='random world-frame 3D direction per force cycle',
            schedule='random rest/ramp-up/hold/ramp-down; zero_prob from task config',
        ) if args.random_foot_force else None,
        video_enabled=args.video,
        external_force_visualized=bool(args.video and args.random_foot_force),
        rng_seed_timing='after checkpoint loading',
        repeats=1, settle_s=args.settle, reach_s=args.reach, hold_s=args.hold,
        protocol='continuous_nearest_targets_v4', continue_on_instability=args.continue_on_instability, default_targets='nearest unvisited feasible point from anchor; full workspace, not validation-only',
        contact_force_measurements=dict(
            left_hand_wall=('filtered ContactSensor normal force, world-frame xyz and magnitude, N'
                            if has_wall else 'not applicable: evaluation scene has no wall'),
            right_support_foot='unfiltered ContactSensor net normal force on right_ankle_roll_link; world-frame xyz and magnitude, N',
            force_sensor_sources=('left_hand_mimic filtered to wall object; right_ankle_roll_link net contact force (ground is its support contact)'
                                  if has_wall else 'right_ankle_roll_link net contact force (ground is its support contact); no wall sensor'),
        ),
        scene='wall' if has_wall else 'no_wall',
        failure='nonfinite state, root displacement >0.20m, gravity z >-0.7, support drift >0.12m'
            + (' or hand drift >0.15m' if has_wall else ' (hand drift is recorded but not a failure criterion without a wall)')), indent=2))
    del state
    app = AppLauncher(headless=True, enable_cameras=cfg.eval_render, device='cuda:0').app
    writer = viewer = env = None
    try:
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply, quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from tensordict import TensorDict
        from PIL import Image, ImageDraw
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); agent.eval()
        # Seed after network construction/loading so checkpoint architecture
        # differences do not change the evaluation random sequence.
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        right_foot_sensor = base.scene['contact_forces']
        right_foot_sensor_ids, right_foot_sensor_names = right_foot_sensor.find_bodies('right_ankle_roll_link')
        if len(right_foot_sensor_ids) != 1:
            raise RuntimeError(
                'Expected one right_ankle_roll_link in contact_forces sensor; '
                f'found {right_foot_sensor_names}'
            )
        right_foot_sensor_id = right_foot_sensor_ids[0]
        cmd.zero_init_prob = 1.
        policy = agent.get_rollout_policy('eval')
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        dt = base.step_dt
        marker = actual_marker = force_arrow = None
        if cfg.eval_render:
            from pxr import UsdGeom, UsdLux, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            # A dedicated render camera avoids headless viewport pose overrides.
            import omni.replicator.core as rep
            camera = UsdGeom.Camera.Define(stage, '/World/FootEvalCamera')
            camera.CreateFocalLengthAttr(24.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateVerticalApertureAttr(27.)
            camera_pose = camera.AddTransformOp()
            base._rgb_annotator.detach()
            base._render_product = rep.create.render_product('/World/FootEvalCamera', (960, 720))
            base._rgb_annotator.attach([base._render_product])
            for path in ('/World/light_0', '/World/light_1'):
                light = UsdLux.DistantLight.Get(stage, path)
                light.GetIntensityAttr().Set(600.)
            def sphere(path, color, radius):
                prim = UsdGeom.Sphere.Define(stage, path)
                prim.CreateRadiusAttr(radius); prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                return UsdGeom.XformCommonAPI(prim)
            marker = sphere('/World/FootTarget', (1., .05, .05), .035)
            actual_marker = sphere('/World/FootActual', (.05, 1., .25), .035)
            force_arrow = UsdGeom.BasisCurves.Define(stage, '/World/ExternalFootForce')
            force_arrow.CreateTypeAttr(UsdGeom.Tokens.linear)
            force_arrow.CreateCurveVertexCountsAttr([2, 2, 2])
            force_arrow.CreateWidthsAttr([.012])
            force_arrow.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            force_arrow.CreateDisplayColorAttr([Gf.Vec3f(1., .72, .05)])
            force_arrow.CreatePointsAttr([Gf.Vec3f(0., 0., 0.)] * 6)
            force_arrow.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            base.sim.set_camera_view(cfg.task.viewer.eye, cfg.task.viewer.lookat)
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output / 'evaluation.mp4'), fps=10, codec='libx264', macro_block_size=2)
        if args.web:
            from scripts.foot_reach.web import Viewer
            viewer = Viewer(args.host, args.port)
            print(f'WEB_READY http://{args.host}:{args.port} (forward this port in VS Code)', flush=True)
        is_init = True
        reference = {}
        def reset():
            nonlocal is_init, reference
            env.reset()
            base.episode_length_buf.zero_()
            cmd.set_target_foot_pos_b(anchor)
            reference = dict(root=cmd.asset.data.root_pos_w.clone(),
                support=cmd.asset.data.body_pos_w[:, cmd.right_asset].clone(),
                hand=cmd.asset.data.body_pos_w[:, cmd.hand_asset].clone())
            is_init = True
            if cfg.eval_render:
                centre = reference['root'][0].cpu().numpy() + np.array([0., 0., -.05])
                eye = centre + np.array([1.8, -2.1, .65])
                view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*centre), Gf.Vec3d(0, 0, 1))
                camera_pose.Set(view.GetInverse())

        def measure():
            data = cmd.asset.data
            sole = data.body_pos_w[:, cmd.left_asset] + quat_apply(data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(1, -1))
            actual = quat_apply_inverse(data.root_quat_w, sole - data.root_pos_w)[0]
            target = cmd.target_foot_pos_b()[0]
            root_drift = torch.linalg.vector_norm(data.root_pos_w - reference['root']).item()
            support_drift = torch.linalg.vector_norm(data.body_pos_w[:, cmd.right_asset] - reference['support']).item()
            hand_drift = torch.linalg.vector_norm(data.body_pos_w[:, cmd.hand_asset] - reference['hand']).item()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(data.root_state_w).all())
            reasons = []
            if not finite: reasons.append('nonfinite')
            if root_drift > .20: reasons.append('root_drift')
            if data.projected_gravity_b[0, 2].item() > -.7: reasons.append('tilt')
            if support_drift > .12: reasons.append('support_drift')
            if has_wall and hand_drift > .15: reasons.append('hand_drift')

            def filtered_contact_force(sensor_name):
                sensor_data = base.scene[sensor_name].data
                normal = sensor_data.force_matrix_w
                if normal is None:
                    raise RuntimeError(f'{sensor_name} has no filtered force_matrix_w data')
                force = torch.nan_to_num(normal[0], nan=0.0, posinf=0.0, neginf=0.0)
                # One sensed robot body and one filtered partner; sum any
                # contacts across their shapes into a world-frame normal force.
                return force.reshape(-1, 3).sum(dim=0)

            hand_wall_force = (filtered_contact_force('eval_left_hand_wall_force') if has_wall
                               else torch.zeros(3, device=base.device))
            contact_data = right_foot_sensor.data
            support_forces = contact_data.net_forces_w
            if support_forces is None:
                history = contact_data.net_forces_w_history
                if history is None:
                    raise RuntimeError('contact_forces sensor has no net contact-force data')
                support_forces = history[:, 0]
            support_foot_force = torch.nan_to_num(
                support_forces[0, right_foot_sensor_id], nan=0.0, posinf=0.0, neginf=0.0)
            external_foot_force = cmd.foot_force_ramp.force[0]
            return dict(actual=actual.cpu().numpy(), target=target.cpu().numpy(),
                error=float(torch.linalg.vector_norm(actual-target)), root_drift=root_drift,
                support_drift=support_drift, hand_drift=hand_drift,
                hand_wall_force=hand_wall_force.cpu().numpy(),
                support_foot_force=support_foot_force.cpu().numpy(),
                external_foot_force=external_foot_force.cpu().numpy(),
                failure='+'.join(reasons)), sole

        def step(target):
            nonlocal is_init
            cmd.set_target_foot_pos_b(target)
            # Refresh the raw observation AFTER changing the command, then apply
            # the checkpoint normalizer exactly once. Student sees policy only.
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            normalized = normalizer(raw)
            carry = normalized.select('policy')
            carry['is_init'] = torch.full((1,1), is_init, dtype=torch.bool, device=base.device)
            carry = policy(carry)
            base.step(carry)  # no automatic resets, no changing evaluation targets
            is_init = False
            return measure()

        def render_frame(m, sole, label):
            if not cfg.eval_render: return None
            target_world = cmd.target_sole_world()[0].cpu().numpy()
            sole_world = sole[0].cpu().numpy().astype(float)
            marker.SetTranslate(Gf.Vec3d(*target_world.astype(float)))
            # Lift the green sphere slightly above the sole surface so it is
            # not hidden by the robot mesh when target and actual are close.
            actual_marker.SetTranslate(Gf.Vec3d(*(sole_world + np.array([0., 0., .035]))))
            force_world = m['external_foot_force'].astype(float)
            force_magnitude = float(np.linalg.norm(force_world))
            if force_magnitude > 1e-4:
                direction = force_world / force_magnitude
                arrow_length = min(.36, max(.12, .012 * force_magnitude))
                head_length = min(.07, max(.025, .28 * arrow_length))
                tip = sole_world + arrow_length * direction
                head_base = tip - head_length * direction
                reference_axis = np.array([0., 0., 1.])
                if abs(float(np.dot(direction, reference_axis))) > .9:
                    reference_axis = np.array([0., 1., 0.])
                side = np.cross(direction, reference_axis)
                side /= max(float(np.linalg.norm(side)), 1e-8)
                head_spread = .45 * head_length
                points = [sole_world, head_base,
                          head_base + head_spread * side, tip,
                          head_base - head_spread * side, tip]
                force_arrow.GetPointsAttr().Set([Gf.Vec3f(*p.tolist()) for p in points])
                force_arrow.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            else:
                force_arrow.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            rgb = base.render('rgb_array')
            if rgb.size == 0: return None
            frame = Image.fromarray(rgb.copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,102), fill=(15,23,42))
            draw.text((12,7), f'{label} | target: red; actual foot: green; external force: yellow arrow', fill='white')
            draw.text((12,30), f'Live-root target xyz (m): {np.round(m["target"],3)} | error: {m["error"]*1000:.1f} mm', fill='white')
            draw.text((12,53), f'External foot force: {force_magnitude:.2f} N', fill='white')
            draw.text((12,76), f'Failure: {m["failure"] or "none"}', fill='white')
            # Draw a screen-space arrow over the *projected force application
            # point* on the sole. This remains visible even if Hydra does not
            # render the 3D curve or the robot occludes it. Its direction is
            # the world force projected through the fixed evaluation camera.
            hud_color = (255, 190, 20)
            if force_magnitude > 1e-4:
                camera_centre = reference['root'][0].cpu().numpy() + np.array([0., 0., -.05])
                camera_eye = camera_centre + np.array([1.8, -2.1, .65])
                forward = camera_centre - camera_eye
                forward /= np.linalg.norm(forward)
                camera_right = np.cross(forward, np.array([0., 0., 1.]))
                camera_right /= np.linalg.norm(camera_right)
                camera_up = np.cross(camera_right, forward)
                focal_px = 960. * 24. / 36.
                def project_to_image(point):
                    relative = point - camera_eye
                    depth = float(np.dot(relative, forward))
                    if depth <= 1e-6:
                        return None
                    return np.array([480. + focal_px * np.dot(relative, camera_right) / depth,
                                     360. - focal_px * np.dot(relative, camera_up) / depth])
                arrow_length = min(.36, max(.12, .012 * force_magnitude))
                image_start = project_to_image(sole_world)
                image_tip = project_to_image(sole_world + arrow_length * force_world / force_magnitude)
                if image_start is not None and image_tip is not None:
                    screen_delta = image_tip - image_start
                    screen_norm = float(np.linalg.norm(screen_delta))
                    if screen_norm < 8.:
                        screen_delta = np.array([0., -18.])
                        screen_norm = 18.
                    screen_dir = screen_delta / screen_norm
                    image_tip = image_start + screen_dir * max(screen_norm, 18.)
                    # Dark outline keeps the yellow arrow legible against the
                    # white robot, then the arrowhead makes its origin clear.
                    draw.line((tuple(image_start), tuple(image_tip)), fill=(20, 20, 20), width=10)
                    draw.line((tuple(image_start), tuple(image_tip)), fill=hud_color, width=6)
                    theta = np.arctan2(screen_dir[1], screen_dir[0])
                    head_len = 16.
                    head = [tuple(image_tip),
                            tuple(image_tip - head_len * np.array([np.cos(theta - .55), np.sin(theta - .55)])),
                            tuple(image_tip - head_len * np.array([np.cos(theta + .55), np.sin(theta + .55)]))]
                    draw.polygon(head, fill=(20, 20, 20))
                    inner = [head[0],
                             tuple(image_tip - 11 * np.array([np.cos(theta - .55), np.sin(theta - .55)])),
                             tuple(image_tip - 11 * np.array([np.cos(theta + .55), np.sin(theta + .55)]))]
                    draw.polygon(inner, fill=hud_color)
                fx, fy, fz = force_world
                draw.text((650,53), f'Force at foot | World F=({fx:.1f},{fy:.1f},{fz:.1f}) N', fill=hud_color)
            else:
                draw.text((790,40), 'NO EXTERNAL FORCE', fill=(180, 180, 180))
            if writer: writer.append_data(np.array(frame))
            if viewer:
                buffer=io.BytesIO(); frame.save(buffer,format='JPEG',quality=80)
                viewer.publish(buffer.getvalue(), dict(target_xyz_m=np.round(m['target'],4).tolist(),
                    actual_xyz_m=np.round(m['actual'],4).tolist(), error_mm=round(m['error']*1000,2),
                    external_foot_force_world_N=np.round(m['external_foot_force'],3).tolist(),
                    failure=m['failure'], status=label))
            return frame

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            reset()
            if args.web:
                desired = anchor.copy(); current = anchor.copy(); paused = False
                start = time.monotonic(); iteration = 0
                m, sole = measure()
                while app.is_running() and (args.web_seconds <= 0 or time.monotonic()-start < args.web_seconds):
                    tick = time.monotonic()
                    for command in viewer.take_commands():
                        if command == 'reset':
                            reset(); desired=anchor.copy(); current=anchor.copy(); paused=False
                            m,sole=measure()
                        elif command == 'pause': paused = not paused
                        else:
                            axis='xyz'.index(command[0]); desired[axis]+=.01*(1 if command[1]=='+' else -1)
                            desired=np.clip(desired,lower,upper)
                    if not paused:
                        delta=desired-current; distance=np.linalg.norm(delta)
                        current+=delta*min(1., .1*dt/max(distance,1e-9))
                        m,sole=step(current)
                        if m['failure']: paused=True
                    if iteration % 5 == 0:
                        render_frame(m,sole,'paused' if paused else 'teleop')
                    iteration+=1
                    time.sleep(max(0.,dt-(time.monotonic()-tick)))
            else:
                summaries=[]
                with (args.output/'samples.csv').open('w', newline='') as f:
                    fields=['point','repeat','time_s','phase','target_x','target_y','target_z',
                            'actual_x','actual_y','actual_z','error_m','root_drift_m','support_drift_m','hand_drift_m',
                            'left_hand_wall_force_x_N','left_hand_wall_force_y_N','left_hand_wall_force_z_N','left_hand_wall_force_norm_N',
                            'right_support_foot_force_x_N','right_support_foot_force_y_N','right_support_foot_force_z_N','right_support_foot_force_norm_N',
                            'external_left_foot_force_x_N','external_left_foot_force_y_N','external_left_foot_force_z_N','external_left_foot_force_norm_N',
                            'failure']
                    csv_writer=csv.DictWriter(f,fieldnames=fields);csv_writer.writeheader()
                    total=0
                    for point,target in enumerate(targets):
                        repeat=0; errors=[]; vectors=[]; failure=''
                        phases=([('settle',args.settle)] if point == 0 else []) + [('reach',args.reach),('hold',args.hold)]
                        for phase,duration in phases:
                            count=max(1,round(duration/dt))
                            for k in range(count):
                                command=anchor if phase=='settle' else target
                                m,sole=step(command)
                                csv_writer.writerow(dict(point=point,repeat=repeat,time_s=total*dt,phase=phase,
                                    **dict(zip(['target_x','target_y','target_z'],m['target'])),
                                    **dict(zip(['actual_x','actual_y','actual_z'],m['actual'])),error_m=m['error'],
                                    root_drift_m=m['root_drift'],support_drift_m=m['support_drift'],hand_drift_m=m['hand_drift'],
                                    **dict(zip(['left_hand_wall_force_x_N','left_hand_wall_force_y_N','left_hand_wall_force_z_N'],m['hand_wall_force'])),
                                    left_hand_wall_force_norm_N=float(np.linalg.norm(m['hand_wall_force'])),
                                    **dict(zip(['right_support_foot_force_x_N','right_support_foot_force_y_N','right_support_foot_force_z_N'],m['support_foot_force'])),
                                    right_support_foot_force_norm_N=float(np.linalg.norm(m['support_foot_force'])),
                                    **dict(zip(['external_left_foot_force_x_N','external_left_foot_force_y_N','external_left_foot_force_z_N'],m['external_foot_force'])),
                                    external_left_foot_force_norm_N=float(np.linalg.norm(m['external_foot_force'])),failure=m['failure']))
                                if total%5==0: render_frame(m,sole,f'point {point}, repeat {repeat}, {phase}')
                                total+=1
                                if phase=='hold' and np.isfinite(m['error']):
                                    errors.append(m['error']); vectors.append(m['actual']-m['target'])
                                if m['failure']:
                                    failure='+'.join(sorted(set(failure.split('+') + m['failure'].split('+')) - {''}))
                                    if not args.continue_on_instability: break
                            if failure and not args.continue_on_instability: break
                        successful=not bool(failure)
                        result=dict(point=point,repeat=repeat,target_root_xyz_m=target.tolist(),
                            failed=bool(failure),failure=failure,completed_hold=len(errors)==max(1,round(args.hold/dt)),success=successful,
                            hold_samples=len(errors),mean_error_m=None,rmse_m=None,p95_error_m=None,max_error_m=None,
                            axis_mae_m=None)
                        if errors:
                            result.update(mean_error_m=float(np.mean(errors)),rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                                p95_error_m=float(np.percentile(errors,95)),max_error_m=float(np.max(errors)),
                                axis_mae_m=np.mean(np.abs(vectors),axis=0).tolist())
                        summaries.append(result)
                        write_reports(args.output, summaries, len(targets))
                        f.flush(); print(json.dumps(result),flush=True)
                        if failure and not args.continue_on_instability:
                            print('Sequence stopped after instability; remaining targets were not attempted.', flush=True)
                            break
                plot_contact_forces(args.output)
                print(f'CONTACT_FORCE_PLOT {args.output / "contact_forces.png"}',flush=True)
                print(f'EVAL_COMPLETE {args.output}',flush=True)
    except KeyboardInterrupt:
        print('Evaluation stopped by user.', flush=True)
    except Exception:
        # Isaac's application shutdown may exit the interpreter before an
        # exception propagates. Report it first and return a nonzero status.
        import traceback
        traceback.print_exc()
        if writer: writer.close()
        if viewer: viewer.close()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        if writer: writer.close()
        if viewer: viewer.close()
        from isaaclab.sim import SimulationContext
        context=SimulationContext.instance()
        if context: context._disable_app_control_on_stop_handle=True
        if env: env.close()
        app.close()


if __name__ == '__main__':
    main()
