"""Replay a crouch reference using the root3 teacher's original online forces."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, json, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--motion-id', type=int, default=311,
               help='Motion ID in crouch_lowlevel/train; 311 is BONES squat_001__A359.csv')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
OmegaConf.set_struct(cfg, False)
OmegaConf.set_readonly(cfg, False)
cfg.checkpoint_path = str(a.checkpoint.resolve())
cfg.vecnorm = 'eval'
cfg.seed = 20261102
cfg.task.num_envs = 1
cfg.task.max_episode_length = 1000
cfg.task.command.guidance_enabled = True
cfg.task.command.dataset.mem_paths = ['crouch_lowlevel/train']
cfg.task.command.dataset.path_weights = [1.0]
cfg.task.command.dataset.fix_ds = 0
cfg.task.command.dataset.fix_motion_id = a.motion_id
cfg.task.command.dataset.resample_on_reset = False
cfg.task.command.dataset.sample_once = True
cfg.wandb.mode = 'disabled'
cfg.eval_render = True
cfg.headless = True
cfg.task.viewer.resolution = [1280, 720]
del state

app = AppLauncher(headless=True, device='cuda:0', enable_cameras=True,
                  kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/omni.tbb.globalcontrol/maxThreadCount=16').app
try:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    from isaacsim.core.utils.viewports import set_camera_view
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.utils.helpers import make_env_policy

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager; robot = cmd.asset; manager = env.action_manager
    if cfg.algo.get('phase') != 'teacher':
        raise RuntimeError(f"Expected teacher checkpoint, got phase={cfg.algo.get('phase')}")
    if list(cmd.point_names) != ['pelvis', 'left_hand_mimic', 'right_hand_mimic']:
        raise RuntimeError(f'Unexpected force points: {cmd.point_names}')
    cmd.zero_init_prob = 1.0
    policy = agent.get_rollout_policy('deploy')
    if list(policy.in_keys) != ['force_gt_', 'policy']:
        raise RuntimeError(f'Unexpected teacher inputs: {policy.in_keys}')
    td = env.reset()
    selected = int(cmd._motion.motion_id[0, 0])
    if selected != a.motion_id:
        raise RuntimeError(f'Requested motion {a.motion_id}, got {selected}')
    source_manifest = json.loads((ROOT / 'dataset/crouch_lowlevel/train/source_manifest.json').read_text())
    source = source_manifest[selected]
    motion_len = int(cmd.lengths[0].item())
    if motion_len < 100:
        raise RuntimeError(f'Crouch clip unexpectedly short: {motion_len} frames')

    body_names = list(cmd.point_names)
    body_ids = cmd.point_ids
    joint_names = list(robot.joint_names)
    dataset_joint_names = list(cmd.dataset.joint_names)
    knee_names = ['left_knee_joint', 'right_knee_joint']
    knee_asset_ids = torch.tensor([joint_names.index(x) for x in knee_names], device=env.device)
    knee_motion_ids = torch.tensor([dataset_joint_names.index(x) for x in knee_names], device=env.device)
    fps, dt = 25, float(env.step_dt)
    total_steps = min(motion_len, 1000)
    writer = imageio.get_writer(a.output / 'rollout.mp4', fps=fps, codec='libx264', quality=8, macro_block_size=1)
    stage = omni.usd.get_context().get_stage()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)

    def project(world_point):
        camera = UsdGeom.Camera(stage.GetPrimAtPath('/OmniverseKit_Persp')).GetCamera(Usd.TimeCode.Default())
        vp = camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix()
        v = Gf.Vec4d(float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0) * vp
        if abs(float(v[3])) < 1e-8:
            return np.array([-100.0, -100.0])
        return np.array([(float(v[0] / v[3]) * .5 + .5) * 1280,
                         (.5 - float(v[1] / v[3]) * .5) * 720])

    def draw_arrow(draw, start3, force3, color):
        mag = float(np.linalg.norm(force3))
        if mag < .25:
            return
        end3 = start3 + force3 / mag * min(.85, .04 * mag)
        start, end = project(start3), project(end3)
        delta = end - start; length = float(np.linalg.norm(delta))
        if length < 3:
            return
        u = delta / length; n = np.array([-u[1], u[0]])
        draw.line([tuple(start), tuple(end)], fill=color, width=6)
        draw.polygon([tuple(end), tuple(end-u*18+n*8), tuple(end-u*18-n*8)], fill=color)
        draw.ellipse((start[0]-5, start[1]-5, start[0]+5, start[1]+5), fill=color)

    rows = []
    fall_time = None
    max_force_by_point = np.zeros(3, dtype=np.float64)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for step in range(total_steps):
            # The teacher sees the same GT force features used in its training rollout.
            observed_gt = td['force_gt_'].reshape(1, 3, 3) * 30.0
            qroot = robot.data.root_quat_w[:, None].expand(-1, 3, -1)
            from active_adaptation.utils.math import quat_apply
            observed_world = quat_apply(qroot, observed_gt)
            torch.testing.assert_close(observed_world[0], cmd.point_force_buffer[0], atol=2e-4, rtol=2e-4)

            td = policy(td)
            td = env.step(td)['next']

            root_w = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            root_target = cmd._motion.root_pos_w[0, 0].detach().cpu().numpy().copy()
            actual_knees = robot.data.joint_pos[0, knee_asset_ids].detach().cpu().numpy().copy()
            target_knees = cmd._motion.joint_pos[0, 0, knee_motion_ids].detach().cpu().numpy().copy()
            forces = cmd.point_force_buffer[0].detach().cpu().numpy().copy()
            positions = robot.data.body_pos_w[0, body_ids].detach().cpu().numpy().copy()
            max_force_by_point = np.maximum(max_force_by_point, np.linalg.norm(forces, axis=-1))
            t = (step + 1) * dt
            height = float(root_w[2] - env.scene.env_origins[0, 2].item())
            height_target = float(root_target[2])
            tilt = float(torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
            if fall_time is None and (height < .45 or tilt > np.pi/3):
                fall_time = t
            frame_idx = int(cmd._motion.step[0, 0].item())
            row = {
                'time_s': t, 'motion_step': frame_idx,
                'root_z_m': height, 'target_root_z_m': height_target,
                'left_knee_rad': float(actual_knees[0]), 'right_knee_rad': float(actual_knees[1]),
                'target_left_knee_rad': float(target_knees[0]), 'target_right_knee_rad': float(target_knees[1]),
                'pelvis_fx_N': float(forces[0,0]), 'pelvis_fy_N': float(forces[0,1]), 'pelvis_fz_N': float(forces[0,2]),
                'left_hand_fx_N': float(forces[1,0]), 'left_hand_fy_N': float(forces[1,1]), 'left_hand_fz_N': float(forces[1,2]),
                'right_hand_fx_N': float(forces[2,0]), 'right_hand_fy_N': float(forces[2,1]), 'right_hand_fz_N': float(forces[2,2]),
            }
            rows.append(row)

            if step % 2 == 0 or step == total_steps - 1:
                center = root_w.copy(); center[2] += .18
                set_camera_view((center + np.array([2.8, 0.0, .15])).tolist(), (center + np.array([0., 0., .05])).tolist())
                image = Image.fromarray(env.render('rgb_array'))
                draw = ImageDraw.Draw(image, 'RGBA')
                for i, color in enumerate([(255, 190, 35), (255, 115, 35), (40, 215, 255)]):
                    draw_arrow(draw, positions[i], forces[i], color)
                draw.rectangle((0, 0, 1280, 144), fill=(8, 14, 23, 225))
                draw.text((16, 8), 'ROOT3 TEACHER • BONES SQUAT REFERENCE • ONLINE JACOBIAN FORCES', font=small, fill=(190, 225, 255))
                draw.text((16, 33), f"motion {selected}: {Path(source.get('source_name', 'AMASS crouch')).name}   |   {t:.2f}/{total_steps*dt:.2f} s", font=font, fill='white')
                draw.text((16, 66), f"root height actual / reference: {height:.3f} / {height_target:.3f} m", font=small, fill='white')
                draw.text((16, 91), f"knees actual: L {actual_knees[0]:.2f}, R {actual_knees[1]:.2f} rad   |   reference: L {target_knees[0]:.2f}, R {target_knees[1]:.2f}", font=small, fill='white')
                force_line = '   |   '.join(f'{body_names[i]}: {np.linalg.norm(forces[i]):.1f} N (Fz {forces[i,2]:+.1f})' for i in range(3))
                draw.text((16, 117), force_line, font=small, fill=(255, 220, 130))
                writer.append_data(np.asarray(image))

    writer.close()
    with (a.output / 'force_and_motion_trace.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    actual_knees = np.array([[r['left_knee_rad'], r['right_knee_rad']] for r in rows])
    target_knees = np.array([[r['target_left_knee_rad'], r['target_right_knee_rad']] for r in rows])
    heights = np.array([r['root_z_m'] for r in rows]); target_heights = np.array([r['target_root_z_m'] for r in rows])
    report = {
        'checkpoint': str(a.checkpoint.resolve()), 'phase': 'teacher',
        'dataset': 'crouch_lowlevel/train', 'motion_id': selected, 'source': source,
        'motion_frames': total_steps, 'duration_s': total_steps * dt, 'fps': fps,
        'force_generation': 'Original OnlineEstimatedComplianceCommand / JacobianComplianceCommand, unchanged from teacher training',
        'force_points': body_names, 'max_applied_force_N_by_point': dict(zip(body_names, max_force_by_point.tolist())),
        'root_height_start_m': float(heights[0]), 'root_height_min_m': float(heights.min()),
        'target_root_height_start_m': float(target_heights[0]), 'target_root_height_min_m': float(target_heights.min()),
        'max_knee_rad_actual': dict(zip(knee_names, actual_knees.max(axis=0).tolist())),
        'max_knee_rad_reference': dict(zip(knee_names, target_knees.max(axis=0).tolist())),
        'knee_tracking_mae_rad': float(np.abs(actual_knees-target_knees).mean()),
        'first_fall_s': fall_time, 'video': 'rollout.mp4', 'trace': 'force_and_motion_trace.csv',
    }
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print('CROUCH_MOTION_FORCE_REPLAY_COMPLETE', json.dumps(report), flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
