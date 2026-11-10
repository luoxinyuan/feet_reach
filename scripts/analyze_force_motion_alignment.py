"""Roll out matched motion clips with the 3- or 9-point GT-force teacher.

Records target joint increments, teacher actions, generated/applied point forces,
and Jacobian reconstruction quality for checking force/motion correspondence.
"""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, csv, json, argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--label', required=True)
p.add_argument('--motions', type=int, nargs='+', default=[210, 4322, 121])
p.add_argument('--max-steps', type=int, default=300,
               help='Limit each sample to this many 50 Hz control steps.')
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
OmegaConf.set_struct(cfg, False); OmegaConf.set_readonly(cfg, False)
cfg.checkpoint_path = str(a.checkpoint.resolve())
cfg.vecnorm = 'eval'; cfg.seed = 20261102; cfg.task.num_envs = 1
cfg.task.max_episode_length = 1200
cfg.task.command.guidance_enabled = True
# Use one common corpus in both conditions so motion IDs and references match.
cfg.task.command.dataset.mem_paths = ['passive_whole_body_full/train']
cfg.task.command.dataset.path_weights = [1.0]
cfg.task.command.dataset.fix_ds = 0
cfg.task.command.dataset.fix_motion_id = int(a.motions[0])
cfg.task.command.dataset.sample_once = True
cfg.task.command.dataset.resample_on_reset = False
cfg.task.termination = {}
cfg.wandb.mode = 'disabled'; cfg.eval_render = False; cfg.headless = True
del state

app = AppLauncher(headless=True, device='cuda:0', enable_cameras=False,
                  kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/omni.tbb.globalcontrol/maxThreadCount=16').app
try:
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.utils.helpers import make_env_policy
    from active_adaptation.utils.jacobian_guidance import reconstruction
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager; robot = cmd.asset
    if not hasattr(cmd, 'point_force_buffer'):
        raise RuntimeError('Command does not expose point_force_buffer')
    if cfg.algo.phase != 'teacher':
        raise RuntimeError(f'Expected GT-force teacher phase; got {cfg.algo.phase}')
    cmd.zero_init_prob = 1.0
    policy = agent.get_rollout_policy('deploy')
    if list(policy.in_keys) != ['force_gt_', 'policy']:
        raise RuntimeError(f'Expected GT-force teacher, got policy inputs {policy.in_keys}')
    body_names = list(cmd.point_names)
    point_fields = [re.sub(r'[^A-Za-z0-9_]+', '_', x) for x in body_names]
    joint_names = list(robot.joint_names)
    motion_joint_names = list(cmd.dataset.joint_names)
    matched_asset, matched_motion = [], []
    for i, name in enumerate(joint_names):
        if name in motion_joint_names:
            matched_asset.append(i); matched_motion.append(motion_joint_names.index(name))
    matched_asset = torch.tensor(matched_asset, device=env.device)
    matched_motion = torch.tensor(matched_motion, device=env.device)
    rows=[]; runs=[]
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for motion_id in a.motions:
            cmd.dataset.fix_motion_id = int(motion_id)
            cmd.dataset._populate_buffer_full(target='A')
            td = env.reset()
            selected = int(cmd._motion.motion_id[0,0])
            length = int(cmd.lengths[0].item())
            nsteps = min(length, cfg.task.max_episode_length-1, a.max_steps)
            run_start = len(rows)
            for step in range(nsteps):
                qtarget = cmd._motion.joint_pos[0,0,matched_motion].detach().clone()
                qtarget_next_motion = cmd.dataset.get_slice(
                    None, torch.minimum(cmd.t+1, cmd.lengths-1), 1)
                qtarget_next = qtarget_next_motion.joint_pos[0,0,matched_motion].detach().clone()
                dq_target = qtarget_next-qtarget
                root_target = cmd._motion.root_pos_w[0,0].detach().clone()
                root_target_next = qtarget_next_motion.root_pos_w[0,0].detach().clone()
                root_actual = robot.data.root_pos_w[0].detach().clone()-env.scene.env_origins[0]
                fd, td_yaw = cmd._intent()
                action_td = policy(td)
                action = action_td['action'][0].detach().clone()
                action_effective = action.tanh()
                next_td = env.step(action_td)
                td = next_td['next']
                forces = cmd.point_force_buffer[0].detach().clone()
                force_norms = forces.norm(dim=-1)
                net_force = forces.sum(0)
                achieved, resid, relative, cosine = reconstruction(
                    cmd.point_jacobians, cmd.point_force_buffer, cmd.generalized_target,
                    cmd.prompt_weights)
                row = {
                    'label':a.label, 'dataset_index':motion_id, 'motion_id':selected, 'step':step,
                    'time_s':(step+1)*float(env.step_dt), 'target_root_z_m':float(root_target[2]),
                    'actual_root_z_before_m':float(root_actual[2]),
                    'target_root_dz_m':float(root_target_next[2]-root_target[2]),
                    'target_dq_rms_rad':float(dq_target.square().mean().sqrt()),
                    'target_dq_norm_rad':float(dq_target.norm()),
                    'effective_action_rms':float(action_effective.square().mean().sqrt()),
                    'action_target_cosine':float(torch.nn.functional.cosine_similarity(action_effective, dq_target/.04, dim=0)),
                    'force_net_x_N':float(net_force[0]),'force_net_y_N':float(net_force[1]),'force_net_z_N':float(net_force[2]),
                    'force_sum_point_norms_N':float(force_norms.sum()),
                    'force_max_point_norm_N':float(force_norms.max()),
                    'root_intent_x_N':float(fd[0,0]),'root_intent_y_N':float(fd[0,1]),'root_intent_z_N':float(fd[0,2]),
                    'root_intent_yaw_Nm':float(td_yaw[0]),
                    'jacobian_relative_error':float(relative[0]),'jacobian_cosine':float(cosine[0]),
                }
                for j,name in enumerate(point_fields):
                    row[f'{name}_fx_N']=float(forces[j,0]);row[f'{name}_fy_N']=float(forces[j,1]);row[f'{name}_fz_N']=float(forces[j,2])
                    row[f'{name}_norm_N']=float(force_norms[j])
                rows.append(row)
                if bool(next_td['next','done'][0].item()):
                    break
            runs.append({'dataset_index':motion_id,'motion_id':selected,
                         'reference_frames':length,'rollout_steps':len(rows)-run_start,
                         'duration_s':(len(rows)-run_start)*float(env.step_dt)})
    with (a.output/'trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'label':a.label,'checkpoint':str(a.checkpoint.resolve()),'phase':str(cfg.algo.phase),
            'force_points':body_names,'dataset':'passive_whole_body_full/train',
            'runs':runs,'trace':'trace.csv',
            'force_limits_N':{'per_point':30.0,'sum_point_norms':60.0},
            'note':'Actions are deterministic teacher outputs; forces are actual applied force_gt labels.'}
    (a.output/'metadata.json').write_text(json.dumps(report,indent=2))
    print('FORCE_MOTION_ALIGNMENT_COMPLETE',json.dumps(report),flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
