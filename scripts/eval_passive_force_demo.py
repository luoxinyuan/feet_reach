"""Single-robot frontal force demonstration for the segmented CEER student."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--dataset-path', required=True)
parser.add_argument('--guidance-cache', type=Path, required=True)
parser.add_argument('--motion-id', type=int, default=323)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
OmegaConf.set_struct(cfg, False)
cfg.checkpoint_path = str(args.checkpoint.resolve())
cfg.vecnorm = 'eval'
cfg.seed = 20261031
cfg.task.num_envs = 1
cfg.task.max_episode_length = 1000
cfg.task.command.dataset.mem_paths = [args.dataset_path]
cfg.task.command.dataset.path_weights = [1.]
cfg.task.command.dataset.resample_on_reset = False
cfg.task.command.dataset.fix_motion_id = args.motion_id
cfg.task.command.guidance_cache = str(args.guidance_cache.resolve())
cfg.wandb.mode = 'disabled'
cfg.eval_render = True
cfg.headless = True
cfg.task.viewer.resolution = [1280, 720]
cfg.task.viewer.eye = [2.8, 0.0, 0.15]
cfg.task.viewer.lookat = [0.0, 0.0, 0.0]
del state

app = AppLauncher(headless=True, device='cuda:0', enable_cameras=True,
                  kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/omni.tbb.globalcontrol/maxThreadCount=16').app
try:
    from scripts.utils.helpers import make_env_policy
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Usd, UsdGeom, Gf
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio

    torch.manual_seed(20261031); np.random.seed(20261031)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager; asset = cmd.asset
    cmd.zero_init_prob = 1.0
    policy = agent.get_rollout_policy('deploy')
    assert list(policy.in_keys) == ['policy']
    td = env.reset()
    if int(cmd._motion.motion_id[0, 0]) != args.motion_id:
        raise RuntimeError(f'Requested motion {args.motion_id}, got {int(cmd._motion.motion_id[0,0])}')
    if not hasattr(cmd, 'point_force_buffer'):
        raise RuntimeError('This checkpoint does not use segmented point forces')

    # Keep the reference pose at the clip's first frame so changes in the
    # video are caused by the force sequence, not by a moving reference.
    original_before_update = cmd.before_update
    def hold_reference_before_update():
        original_before_update()
        cmd.t.zero_(); cmd.finished.zero_()
        cmd._motion = cmd.dataset.get_slice(None, cmd.t, steps=cmd.future_steps)
        cmd.update_reward_target(); cmd._reference(); cmd._solve()
    cmd.before_update = hold_reference_before_update

    point_names = list(cmd.point_names)
    point_index = point_names.index('left_hand')
    # Force vectors are in world axes; with the robot facing the camera along
    # +X, +/-Y reads as left/right and +/-Z as up/down in the frontal view.
    phases = [
        {'label':'No force', 'force':(0.,0.,0.), 'duration_s':1.0},
        {'label':'Left hand: 5 N to robot-left', 'force':(0.,5.,0.), 'duration_s':2.0},
        {'label':'Left hand: 10 N to robot-left', 'force':(0.,10.,0.), 'duration_s':2.0},
        {'label':'Left hand: 10 N to robot-right', 'force':(0.,-10.,0.), 'duration_s':2.0},
        {'label':'Left hand: 8 N upward', 'force':(0.,0.,8.), 'duration_s':2.0},
        {'label':'Left hand: 8 N downward', 'force':(0.,0.,-8.), 'duration_s':2.0},
        {'label':'Left hand: 12 N forward', 'force':(12.,0.,0.), 'duration_s':2.0},
        {'label':'No force', 'force':(0.,0.,0.), 'duration_s':1.0},
    ]
    dt = float(env.step_dt)
    step_counts = [round(x['duration_s']/dt) for x in phases]
    boundaries = np.cumsum([0] + step_counts)
    total_steps = int(boundaries[-1])

    # Replace cache guidance after each reference solve with this visible,
    # known force sequence. The command's normal slew limiter and point-force
    # application path remain active.
    original_solve = cmd._solve
    def solve_with_schedule():
        original_solve()
        phase = min(int(np.searchsorted(boundaries[1:], step_index, side='right')), len(phases)-1)
        cmd.point_desired.zero_()
        cmd.point_desired[0, point_index] = torch.as_tensor(phases[phase]['force'], device=env.device)
    cmd._solve = solve_with_schedule
    # The custom before_update calls _solve, including at reset; set the first
    # force explicitly so the observation and physics stay consistent.
    step_index = 0
    cmd.point_desired.zero_()

    # Let the renderer's camera settle and the reset pose converge before the
    # recorded sequence begins; warm-up steps are unforced and not in the video.
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for _ in range(50):
            td = policy(td)
            td = env.step(td)['next']
            root_now = asset.data.root_pos_w[0].cpu().numpy()
            env.base_env.cfg.viewer.eye = [2.8, 0.0, 0.15]
            env.base_env.cfg.viewer.lookat = [0.0, 0.0, 0.0]
            set_camera_view((root_now + np.array([2.8, 0.0, 0.15])).tolist(),
                            root_now.tolist())
            env.render('rgb_array')

    root = asset.data.root_pos_w[0].cpu().numpy()
    eye = (root + np.array([2.8, 0.0, 0.15])).tolist()
    look = root.tolist()
    set_camera_view(eye, look)
    stage = __import__('omni.usd', fromlist=['get_context']).get_context().get_stage()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    writer = imageio.get_writer(args.output/'rollout.mp4', fps=25, codec='libx264', quality=8, macro_block_size=1)
    rows=[]; frame_count=0; max_force=0.; fallen_at=None
    colors=[(255,190,35)]
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for step_index in range(total_steps):
            phase_i = min(int(np.searchsorted(boundaries[1:], step_index, side='right')), len(phases)-1)
            td = policy(td)
            td = env.step(td)['next']
            if bool(td['terminated'].any()) and fallen_at is None:
                fallen_at = step_index*dt
            if step_index % 2 == 0 or step_index == total_steps-1:
                # Isaac Lab's render path reads eye/lookat relative to root.
                env.base_env.cfg.viewer.eye = [2.8, 0.0, 0.15]
                env.base_env.cfg.viewer.lookat = [0.0, 0.0, 0.0]
                root_now = asset.data.root_pos_w[0].cpu().numpy()
                set_camera_view((root_now + np.array([2.8, 0.0, 0.15])).tolist(),
                                (root_now + np.array([0.0, 0.0, 0.0])).tolist())
                rgb = env.render('rgb_array')
                im = Image.fromarray(rgb); draw = ImageDraw.Draw(im, 'RGBA')
                camera = UsdGeom.Camera(stage.GetPrimAtPath('/OmniverseKit_Persp')).GetCamera(Usd.TimeCode.Default())
                vp = camera.frustum.ComputeViewMatrix()*camera.frustum.ComputeProjectionMatrix()
                def project(x):
                    v=Gf.Vec4d(float(x[0]),float(x[1]),float(x[2]),1.)*vp
                    if abs(float(v[3]))<1e-8:return np.array([-100.,-100.])
                    return np.array([(float(v[0]/v[3])*.5+.5)*1280,(.5-float(v[1]/v[3])*.5)*720])
                p = cmd.point_positions_w()[0, point_index].cpu().numpy()
                f = cmd.point_force_buffer[0, point_index].cpu().numpy()
                mag = float(np.linalg.norm(f)); max_force=max(max_force,mag)
                if mag > .2:
                    # Deliberately lengthen the overlay arrow while keeping its
                    # tail projected from the actual force application point.
                    a=project(p); b=project(p + f/mag*min(.85,.12*mag)); d=b-a; norm=np.linalg.norm(d)
                    if norm>3:
                        u=d/norm; n=np.array([-u[1],u[0]]); col=colors[0]
                        draw.line([tuple(a),tuple(b)],fill=col,width=8)
                        draw.polygon([tuple(b),tuple(b-u*22+n*12),tuple(b-u*22-n*12)],fill=col)
                        draw.ellipse((a[0]-8,a[1]-8,a[0]+8,a[1]+8),fill=col,outline=(30,20,0),width=2)
                t=step_index*dt
                draw.rectangle((0,0,1280,108),fill=(10,15,24,225))
                draw.text((22,12),'ONE ROBOT • FRONTAL VIEW • SEGMENTED CEER POLICY',font=small,fill=(190,225,255))
                draw.text((22,43),f'{phases[phase_i]["label"]}    |    applied {mag:.1f} N    |    {t:.1f} / {total_steps*dt:.1f} s',font=font,fill='white')
                draw.rectangle((18,120,430,185),fill=(10,15,24,210))
                draw.text((32,131),'Yellow arrow starts at the left hand',font=small,fill=(255,210,80))
                draw.text((32,157),'Arrow direction = force direction',font=small,fill='white')
                writer.append_data(np.asarray(im)); frame_count+=1
            rows.append({'time_s':step_index*dt,'phase':phases[phase_i]['label'],
                         'force_N':cmd.point_force_buffer[0,point_index].cpu().tolist(),
                         'application_point_m':cmd.point_positions_w()[0,point_index].cpu().tolist(),
                         'root_pos_m':asset.data.root_pos_w[0].cpu().tolist(),
                         'left_hand_pos_m':asset.data.body_pos_w[0,asset.body_names.index('left_hand_mimic')].cpu().tolist(),
                         'right_hand_pos_m':asset.data.body_pos_w[0,asset.body_names.index('right_hand_mimic')].cpu().tolist()})
    writer.close()
    summary={'checkpoint':str(args.checkpoint.resolve()),'motion_id':args.motion_id,'seed':20261031,
             'duration_s':total_steps*dt,'video_fps':25,'video_frames':frame_count,
             'camera':'single frontal robot; eye along +X','force_point':'left_hand',
             'max_applied_force_N':max_force,'fall_time_s':fallen_at,
             'phases':phases,'initial_pose_reference_held':True}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'trace.json').write_text(json.dumps(rows))
    print('FORCE_DEMO_COMPLETE',json.dumps(summary),flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle=True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
