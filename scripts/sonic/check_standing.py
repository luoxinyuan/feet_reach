"""Ten-second zero-residual SONIC standing check in the actual training asset."""
import os
os.environ['ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE'] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["sonic", "ceer"], default="sonic")
parser.add_argument("--ceer-force-limit", type=float, default=5.)
args = parser.parse_args()
import math
import torch
import hydra
from tensordict import TensorDict
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device='cuda:0').app
try:
    from active_adaptation.envs import SimpleEnv
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'cfg'), version_base=None):
        cfg = hydra.compose(config_name='train', overrides=[
            f'task=G1/G1_{args.backend}_compliance', f'algo={args.backend}_residual',
            'task.num_envs=8', 'task.command.guidance_enabled=false'])
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    if args.backend == "ceer":
        cfg.task.action.ceer_force_limit = args.ceer_force_limit
    env = SimpleEnv(cfg.task); env.reset()
    robot = env.scene['robot']; manager = env.action_manager
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    ids = torch.arange(env.num_envs, device=env.device)
    if args.backend == "ceer":
        env.command_manager.joint_pos_boot_protect.copy_(robot.data.joint_pos)
    manager.reset(ids)
    env.command_manager.force.zero_(); env.command_manager.desired.zero_()
    td = TensorDict({'action': torch.zeros(env.num_envs, 29, device=env.device)}, [env.num_envs])
    min_z, max_tilt, max_drift = 10., 0., 0.
    with torch.no_grad():
        for _ in range(500):
            # No reward/termination/AMASS reset can conceal a fall during this test.
            env._update_sim(td)
            z = robot.data.root_pos_w[:,2] - env.scene.env_origins[:,2]
            min_z = min(min_z, z.min().item())
            tilt = torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
            max_tilt = max(max_tilt, tilt.max().item())
            max_drift = max(max_drift, (robot.data.root_pos_w[:,:2]-root[:,:2]).norm(dim=-1).max().item())
            assert torch.isfinite(manager.low_action).all()
    fk_error = None
    if args.backend == 'ceer':
        from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude
        body_ids = [robot.body_names.index(side+'_hand_mimic') for side in ('left','right')]
        predicted_p, predicted_q = manager.kinematics.wrists(robot.data.joint_pos[:,manager.joint_ids])
        measured_p, measured_q = subtract_frame_transforms(
            robot.data.root_pos_w[:,None].expand(-1,2,-1),
            robot.data.root_quat_w[:,None].expand(-1,2,-1),
            robot.data.body_pos_w[:,body_ids], robot.data.body_quat_w[:,body_ids])
        fk_error = {'position_m': (predicted_p-measured_p).abs().max().item(),
                    'rotation_rad': quat_error_magnitude(predicted_q,measured_q).max().item()}
        assert fk_error['position_m'] < .002 and fk_error['rotation_rad'] < .01, fk_error
    hold_error = (manager.reference.q-manager.nominal).abs().max().item()
    # Independently exercise nonzero joint-reference residuals through SONIC.
    # This synthetic diagnostic is not a student observation or training target.
    joint = manager.joint_names.index('left_elbow_joint')
    errors = []
    with torch.no_grad():
        for step in range(300):
            desired = manager.nominal.expand(env.num_envs, -1).clone()
            desired[:, joint] += .2 * math.sin(2 * math.pi * (step + 1) * .02 / 4.)
            normalized = (desired - manager.reference.q) / (manager.reference.dt * manager.reference.speed)
            td['action'] = torch.atanh(normalized.clamp(-.999, .999))
            env._update_sim(td)
            errors.append((robot.data.joint_pos[:, manager.joint_ids[joint]] - manager.reference.q[:, joint]).abs().mean().item())
            assert (robot.data.root_pos_w[:,2] - env.scene.env_origins[:,2]).min() > .5
    arm_mae = sum(errors) / len(errors)
    report = {'seconds': 10, 'num_envs': env.num_envs, 'min_root_height_m': min_z,
              'max_tilt_rad': max_tilt, 'max_xy_drift_m': max_drift,
              'joint_reference_hold_error': hold_error, 'sinusoidal_elbow_mae_rad': arm_mae, 'motion_seconds': 6, 'fk_error': fk_error, 'ceer_force_limit': args.ceer_force_limit}
    output = ROOT / f'artifacts/{args.backend}_residual/standing.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)); print('STANDING_RESULT', report, flush=True)
    assert min_z > .5 and max_tilt < .6 and arm_mae < .15, report
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
