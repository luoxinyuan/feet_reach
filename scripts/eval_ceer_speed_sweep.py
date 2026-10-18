"""Measure CEER low-level wall loading for constant-speed EE target approaches."""
import os, sys, json, csv, argparse, hashlib, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))
import numpy as np


def contact_impulse_100ms(rows, threshold_n=1.0, window_s=0.1):
    """Integrate filtered normal force for 100 ms after first sampled contact."""
    times = np.asarray([float(row['time_s']) for row in rows], dtype=float)
    forces = np.asarray([float(row['normal_force_N']) for row in rows], dtype=float)
    contact = np.flatnonzero(forces > threshold_n)
    if not len(contact):
        return None
    start = times[contact[0]]
    end = start + window_s
    if times[-1] < end:
        return None
    inside = (times > start) & (times < end)
    sample_times = np.concatenate(([start], times[inside], [end]))
    sample_forces = np.interp(sample_times, times, forces)
    return float(np.trapz(sample_forces, sample_times))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Residual checkpoint used only to construct the zero-residual CEER environment')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--speeds', type=float, nargs='+', default=[.1, .2, .3, .4, .5, .6])
    p.add_argument('--wall-x', type=float, default=.44)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--video-fps', type=int, default=25)
    p.add_argument('--force-sample-hz', type=float, default=50., help='force and hand-position sample rate; must align with the selected physics rate')
    p.add_argument('--physics-rate-hz', type=float, default=200., help='physics integration rate; increase this to collect real samples above the original 200 Hz rate')
    p.add_argument('--append', action='store_true', help='append new speeds to an existing sweep and regenerate combined plots')
    args = p.parse_args()
    if any(v <= 0 for v in args.speeds) or not 1 <= args.video_fps <= 50:
        p.error('Speeds must be positive and video-fps must be 1..50')
    if args.append:
        if not (args.output / 'summary.json').exists():
            p.error('--append requires an existing completed sweep in the output directory')
        previous_summary = json.loads((args.output / 'summary.json').read_text())
        previous_results = previous_summary.get('results', [])
        previous_config = json.loads((args.output / 'config.json').read_text())
        prior_physics_rate = float(previous_config.get('physics_rate_hz', 200.))
        prior_sample_rate = float(previous_config.get('force_sample_hz', 50.))
        if (previous_config.get('seed') != args.seed or abs(previous_config.get('wall_x_m', -1) - args.wall_x) > 1e-9
                or abs(prior_physics_rate - args.physics_rate_hz) > 1e-9
                or abs(prior_sample_rate - args.force_sample_hz) > 1e-9):
            p.error('Append settings must match the existing seed, wall position, physics rate, and sample rate')
        existing_speeds = {round(float(x['target_speed_m_s']), 6) for x in previous_results}
        if any(round(v, 6) in existing_speeds for v in args.speeds):
            p.error('Append speeds must not duplicate an existing speed')
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        previous_results = []
        args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['policy']['last_phase'] not in ('adapt', 'finetune'):
        raise ValueError('A student-phase proactive-residual checkpoint is required')
    cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
    OmegaConf.set_struct(cfg, False)
    if not cfg.task.command._target_.endswith('ProactiveResidualCommand'):
        raise ValueError('A proactive-residual checkpoint is required')
    if args.physics_rate_hz <= 0:
        raise ValueError('physics-rate-hz must be positive')
    cfg.task.sim.isaac_physics_dt = 1. / args.physics_rate_hz
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.vecnorm = 'eval'; cfg.seed = args.seed
    cfg.task.num_envs = 1; cfg.task.max_episode_length = 10000000; cfg.task.termination = {}
    cfg.task.command.student_train = True; cfg.task.command.disable_motion_finish = True
    cfg.task.command.dataset.fix_ds = 0; cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.eval_contact_forces = True
    cfg.task.viewer.resolution = [640, 480]
    cfg.eval_render = True; cfg.headless = True; cfg.app = dict(headless=True, enable_cameras=True)
    cfg.wandb.mode = 'disabled'
    (args.output / 'config.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        phase=state['policy']['last_phase'], policy='frozen CEER low-level, high-level residual fixed at zero',
        speeds_m_s=sorted([float(x['target_speed_m_s']) for x in previous_results] + args.speeds), wall_x_m=args.wall_x, seed=args.seed,
        path='constant +x world-frame EE target velocity from reset hand position to wall_x+0.035 m; then hold 0.5 s',
        physics_rate_hz=args.physics_rate_hz, physics_dt_s=1. / args.physics_rate_hz, control_dt_s=.02,
        force_source=f'left_hand_mimic filtered to reach_wall; {args.force_sample_hz:g} Hz samples, 200 Hz physics',
        force_sample_hz=args.force_sample_hz,
        contact_threshold_N=1.0, contact_impulse_window_s=0.1,
        contact_impulse_method='Trapezoidal integral of filtered normal force from first sampled force > 1 N through 100 ms later; interpolate window endpoints.',
        video_fps=args.video_fps), indent=2))
    del state

    app = AppLauncher(headless=True, enable_cameras=True, device='cuda:0').app
    writer = None
    try:
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from tensordict import TensorDict
        from PIL import Image, ImageDraw
        import imageio.v2 as imageio
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pxr import UsdGeom, UsdLux, Gf
        import omni.usd, omni.replicator.core as rep

        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); agent.eval()
        base = env.base_env; cmd = base.command_manager
        assert abs(base.physics_dt - 1. / args.physics_rate_hz) < 1e-9 and abs(base.step_dt - .02) < 1e-9
        sample_stride = round(1. / (args.force_sample_hz * base.physics_dt))
        if sample_stride < 1 or abs(1. / (sample_stride * base.physics_dt) - args.force_sample_hz) > 1e-6:
            raise ValueError('force-sample-hz must align exactly with the selected physics rate')
        sample_dt = sample_stride * base.physics_dt
        sensor = base.scene['eval_left_hand_wall_force']
        normalizer = vecnorm.to_observation_norm(); normalizer.eval()
        actor_obs_key = 'hl_policy'
        fresh = True
        rows = []
        active = False
        clock_t = 0.
        previous_p = None
        previous_f = 0.
        sample_counter = 0
        video_stride = max(1, round(1. / (args.video_fps * base.step_dt)))

        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, '/World/CEERSpeedCamera')
        camera.CreateFocalLengthAttr(24.); camera.CreateHorizontalApertureAttr(36.); camera.CreateVerticalApertureAttr(27.)
        eye = np.array([.10, -2.1, 1.22]); centre = np.array([.27, .10, .84])
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*centre), Gf.Vec3d(0, 0, 1)).GetInverse())
        base._rgb_annotator.detach()
        base._render_product = rep.create.render_product('/World/CEERSpeedCamera', (640, 480))
        base._rgb_annotator.attach([base._render_product])
        for path in ('/World/light_0', '/World/light_1'):
            UsdLux.DistantLight.Get(stage, path).GetIntensityAttr().Set(600.)
        markers = []
        for name, color, radius in [('Target', (1., .05, .05), .025), ('Actual', (.05, 1., .2), .018)]:
            prim = UsdGeom.Sphere.Define(stage, '/World/CEERSpeed' + name)
            prim.CreateRadiusAttr(radius); prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            markers.append(UsdGeom.XformCommonAPI(prim))
        forward = (centre - eye) / np.linalg.norm(centre - eye)
        right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        def project(xyz):
            delta = xyz - eye; depth = float(delta @ forward)
            return (320 + 427 * float(delta @ right) / depth, 240 - 427 * float(delta @ up) / depth) if depth > 0 else None

        def physics_sample(_):
            nonlocal clock_t, previous_p, previous_f, sample_counter
            if not active:
                return
            clock_t += base.physics_dt; sample_counter += 1
            if sample_counter % sample_stride:
                return
            data = cmd.asset.data
            pos = data.body_pos_w[0, cmd.hand_asset].detach().cpu().numpy().copy()
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError('Filtered hand-wall force sensor has no force matrix')
            force = matrix[0].reshape(-1, 3).sum(0).detach().cpu().numpy()
            if not np.isfinite(force).all():
                raise RuntimeError('Nonfinite filtered contact force')
            fn = max(0., -float(force[0]))
            vx = (pos[0] - previous_p[0]) / sample_dt if previous_p is not None else 0.
            actual = quat_apply_inverse(data.root_quat_w, data.body_pos_w[:, cmd.hand_asset] - data.root_pos_w)[0].cpu().numpy()
            target = cmd.target_ee_pos_b()[0].cpu().numpy()
            rows.append(dict(time_s=round(clock_t, 6), phase='approach' if clock_t <= approach_s else 'hold',
                actual_world_x=float(pos[0]), target_world_x=float(cmd.target_ee_world()[0, 0]),
                actual_root_x=float(actual[0]), target_root_x=float(target[0]), wall_gap_m=args.wall_x-float(pos[0]),
                vx_m_s=float(vx), normal_force_N=fn, positive_force_rate_N_s=max(0., (fn-previous_f)/sample_dt),
                force_x_N=float(force[0]), force_y_N=float(force[1]), force_z_N=float(force[2]),
                root_height_m=float(data.root_pos_w[0, 2])))
            previous_p = pos; previous_f = fn

        base._post_step_callbacks.append(physics_sample)
        results = list(previous_results)
        with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for speed in args.speeds:
                nonlocal_values = None
                active = False
                rows = []
                env.set_seed(args.seed)
                cmd.cloud_counter = 0; cmd.no_wall_prob = 0.; cmd.wall_range = (args.wall_x, args.wall_x)
                cmd.set_target_ee_pos_b(None); env.reset(); base.episode_length_buf.zero_()
                # Reposition the command at the current EE marker, then move it at the requested constant world-x speed.
                start = cmd.asset.data.body_pos_w[0, cmd.hand_asset].detach().cpu().numpy().astype(float)
                root = cmd.asset.data.root_pos_w[0].detach().cpu().numpy().astype(float)
                q = cmd.asset.data.root_quat_w[0].detach().cpu().numpy()
                start_root = quat_apply_inverse(cmd.asset.data.root_quat_w,
                    cmd.asset.data.body_pos_w[:, cmd.hand_asset] - cmd.asset.data.root_pos_w)[0].detach().cpu().numpy().astype(float)
                goal_x = args.wall_x + .035
                approach_s = max(0., (goal_x - start[0]) / speed)
                start_target_world = start.copy()
                cmd.set_target_ee_pos_b(start_root.reshape(1, 3))
                for _ in range(3): base.sim.render()
                previous_p = start.copy(); previous_f = 0.; clock_t = 0.; sample_counter = 0
                active = True; fresh = True
                directory = args.output / f'speed_{speed:.1f}mps'; directory.mkdir(exist_ok=False)
                writer = imageio.get_writer(str(directory / 'evaluation.mp4'), fps=args.video_fps, codec='libx264', macro_block_size=2)
                total_steps = math.ceil((approach_s + .5) / base.step_dt)
                for i in range(total_steps):
                    t = i * base.step_dt
                    target_world = start_target_world.copy()
                    target_world[0] = min(start[0] + speed * t, goal_x)
                    target_b = quat_apply_inverse(cmd.asset.data.root_quat_w,
                        torch.as_tensor(target_world, dtype=torch.float32, device=base.device).reshape(1, 3) - cmd.asset.data.root_pos_w)
                    cmd.set_target_ee_pos_b(target_b)
                    raw = TensorDict({}, batch_size=[1], device=base.device)
                    base._compute_observation(raw)
                    carry = normalizer(raw).select(actor_obs_key, 'geometry_')
                    carry['is_init'] = torch.tensor([[fresh]], device=base.device)
                    carry['action'] = torch.zeros((1, int(cfg.task.action.command_dim)), dtype=torch.float32, device=base.device)
                    base.step(carry); fresh = False
                    if i % video_stride == 0 and rows:
                        actual_world = cmd.asset.data.body_pos_w[0, cmd.hand_asset].detach().cpu().numpy().astype(float)
                        target_world_draw = cmd.target_ee_world()[0].detach().cpu().numpy().astype(float)
                        for marker, xyz in zip(markers, [target_world_draw, actual_world]): marker.SetTranslate(Gf.Vec3d(*xyz))
                        img = Image.fromarray(base.render('rgb_array').copy()); draw = ImageDraw.Draw(img)
                        for xyz, color in [(target_world_draw, 'red'), (actual_world, 'lime')]:
                            pixel = project(xyz)
                            if pixel:
                                x, y = pixel; draw.ellipse((x-8, y-8, x+8, y+8), outline=color, width=3)
                        r = rows[-1]
                        draw.rectangle((0, 0, 640, 58), fill=(15, 23, 42))
                        draw.text((8, 5), f'CEER zero-residual | target speed {speed:.1f} m/s | red target, green hand', fill='white')
                        draw.text((8, 24), f't={r["time_s"]:.2f}s | hand vx={r["vx_m_s"]:.3f}m/s | filtered wall force={r["normal_force_N"]:.1f}N', fill='white')
                        draw.text((8, 42), f'wall x={args.wall_x:.3f}m | target x={target_world_draw[0]:.3f}m', fill='white')
                        writer.append_data(np.asarray(img))
                    if i and i % 25 == 0:
                        print(f'PROGRESS speed={speed:.1f} step={i}/{total_steps} t={clock_t:.2f}s', flush=True)
                    if float(cmd.asset.data.root_pos_w[0, 2]) < .35:
                        print(f'STOP unstable root height at speed={speed:.1f}', flush=True); break
                writer.close(); writer = None; active = False
                with (directory / 'samples.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                contact_idx = next((i for i, r in enumerate(rows) if r['normal_force_N'] > 1.), None)
                pre = [r for r in rows[:contact_idx] if contact_idx is not None] if contact_idx is not None else [r for r in rows if r['time_s'] <= approach_s]
                force = np.asarray([r['normal_force_N'] for r in rows]); rates = np.asarray([r['positive_force_rate_N_s'] for r in rows])
                onset_rate = rows[contact_idx]['positive_force_rate_N_s'] if contact_idx is not None else None
                result = dict(target_speed_m_s=speed, measured_precontact_speed_m_s=float(np.mean([max(0., r['vx_m_s']) for r in pre])) if pre else None,
                    contact_time_s=rows[contact_idx]['time_s'] if contact_idx is not None else None,
                    first_contact_force_N=rows[contact_idx]['normal_force_N'] if contact_idx is not None else None,
                    first_contact_loading_rate_N_s=onset_rate,
                    peak_force_N=float(force.max()) if len(force) else None,
                    max_positive_loading_rate_N_s=float(rates.max()) if len(rates) else None,
                    final_hold_force_N=float(np.mean([r['normal_force_N'] for r in rows if r['time_s'] >= approach_s])) if any(r['time_s'] >= approach_s for r in rows) else None,
                    contact_impulse_100ms_Ns=contact_impulse_100ms(rows),
                    contact=contact_idx is not None, samples=len(rows), video=str((directory/'evaluation.mp4').relative_to(args.output)), samples_csv=str((directory/'samples.csv').relative_to(args.output)))
                (directory / 'summary.json').write_text(json.dumps(result, indent=2))
                results.append(result)
                print('RESULT', json.dumps(result), flush=True)
        summary = dict(results=results, force_source=f'filtered hand-wall normal force sampled at {args.force_sample_hz:g} Hz',
            contact_impulse='Trapezoidal integral of filtered normal force over 100 ms after the first sampled force > 1 N; window endpoints linearly interpolated; units N*s.',
            loading_rate=f'positive finite difference between adjacent {args.force_sample_hz:g} Hz force samples; same sampling convention across all speeds')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        t0 = None
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            with (args.output / result['samples_csv']).open() as f: rr = list(csv.DictReader(f))
            ax.plot([float(r['time_s']) for r in rr], [float(r['normal_force_N']) for r in rr], label=f"target {result['target_speed_m_s']:.1f} m/s")
        ax.set(xlabel='Time after approach starts (s)', ylabel='Filtered hand-wall normal force (N)',
            title=f'CEER baseline wall force ({args.force_sample_hz:g} Hz samples; {args.physics_rate_hz:g} Hz physics)')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(args.output/'force_curves.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            with (args.output / result['samples_csv']).open() as f: rr = list(csv.DictReader(f))
            ax.plot([float(r['time_s']) for r in rr], [float(r['vx_m_s']) for r in rr], label=f"target {result['target_speed_m_s']:.1f} m/s")
        ax.axhline(0, color='k', lw=.8); ax.set(xlabel='Time after approach starts (s)', ylabel='Measured hand x velocity (m/s)',
            title=f'CEER baseline hand velocity ({args.force_sample_hz:g} Hz samples; {args.physics_rate_hz:g} Hz physics)')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(args.output/'velocity_curves.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        contacts = [r for r in results if r.get('contact') and r.get('contact_impulse_100ms_Ns') is not None]
        ax.plot([r['target_speed_m_s'] for r in contacts], [r['contact_impulse_100ms_Ns'] for r in contacts], marker='o')
        ax.set(xlabel='EE target speed (m/s)', ylabel='100 ms post-contact force impulse (N·s)',
            title=f'CEER baseline contact impulse ({args.force_sample_hz:g} Hz samples; {args.physics_rate_hz:g} Hz physics)')
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(args.output/'contact_impulse_100ms.png', dpi=160); plt.close(fig)
        print('EVAL_COMPLETE', args.output, flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    finally:
        if writer: writer.close()
        app.close()


if __name__ == '__main__':
    main()
