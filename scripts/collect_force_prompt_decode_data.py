"""Collect held-out motion samples for force-prompt -> next-frame delta-q probes."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, json, argparse, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--motion-count', type=int, default=160)
p.add_argument('--steps-per-motion', type=int, default=100)
p.add_argument('--seed', type=int, default=20261103)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
OmegaConf.set_struct(cfg, False); OmegaConf.set_readonly(cfg, False)
cfg.checkpoint_path = str(a.checkpoint.resolve())
cfg.vecnorm = 'eval'; cfg.seed = a.seed; cfg.task.num_envs = 1
cfg.task.max_episode_length = 1200
cfg.task.command.guidance_enabled = True
cfg.task.command.dataset.mem_paths = ['passive_whole_body_full/train']
cfg.task.command.dataset.path_weights = [1.0]
cfg.task.command.dataset.fix_ds = 0
cfg.task.command.dataset.fix_motion_id = None
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
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager; robot = cmd.asset
    # Ensure both checkpoints see the identical frame windows for each motion.
    cmd.zero_init_prob = 1.0
    if cfg.algo.phase != 'teacher':
        raise RuntimeError(f'Expected GT-force teacher phase; got {cfg.algo.phase}')
    policy = agent.get_rollout_policy('deploy')
    if list(policy.in_keys) != ['force_gt_', 'policy']:
        raise RuntimeError(f'Expected GT-force teacher, got policy inputs {policy.in_keys}')
    ds = cmd.dataset.datasets[0]
    total = int(ds.num_motions)
    rng = np.random.default_rng(a.seed)
    motion_indices = np.sort(rng.choice(total, size=min(a.motion_count, total), replace=False))
    joint_names = list(robot.joint_names)
    data_joint_names = list(cmd.dataset.joint_names)
    asset_idx=[]; motion_idx=[]
    for i,n in enumerate(joint_names):
        if n in data_joint_names:
            asset_idx.append(i); motion_idx.append(data_joint_names.index(n))
    asset_idx = torch.tensor(asset_idx, device=env.device)
    motion_idx = torch.tensor(motion_idx, device=env.device)
    obs_rows=[]; force_rows=[]; dq_rows=[]; motion_rows=[]; step_rows=[]
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for num, mid in enumerate(motion_indices):
            cmd.dataset.fix_motion_id = int(mid)
            cmd.dataset._populate_buffer_full(target='A')
            td = env.reset()
            selected = int(cmd._motion.motion_id[0,0])
            length = int(cmd.lengths[0].item())
            nsteps = min(length-1, a.steps_per_motion)
            for step in range(nsteps):
                q0 = cmd._motion.joint_pos[0,0,motion_idx]
                nxt = cmd.dataset.get_slice(None, torch.minimum(cmd.t+1, cmd.lengths-1), 1)
                dq = nxt.joint_pos[0,0,motion_idx] - q0
                obs = td['policy'][0].detach().float().cpu().numpy()
                # Record the exact GT prompt the teacher actor receives now
                # (already root-frame and scaled by the environment observation).
                force = td['force_gt_'][0].detach().float().cpu().numpy()
                action_td = policy(td)
                next_td = env.step(action_td)
                obs_rows.append(obs); force_rows.append(force)
                dq_rows.append(dq.detach().float().cpu().numpy())
                motion_rows.append(int(mid)); step_rows.append(step)
                td = next_td['next']
            if (num+1) % 20 == 0:
                print(f'collected {num+1}/{len(motion_indices)} clips, {len(obs_rows)} transitions', flush=True)
    np.savez_compressed(a.output, state=np.asarray(obs_rows,dtype=np.float32),
        force=np.asarray(force_rows,dtype=np.float32), delta_q=np.asarray(dq_rows,dtype=np.float32),
        motion_index=np.asarray(motion_rows,dtype=np.int32), step=np.asarray(step_rows,dtype=np.int16),
        point_names=np.asarray(cmd.point_names), joint_names=np.asarray([joint_names[i] for i in asset_idx.cpu().tolist()]),
        seed=np.asarray(a.seed), checkpoint=np.asarray(str(a.checkpoint.resolve())))
    print('COLLECTION_COMPLETE', json.dumps({'output':str(a.output),'motions':len(motion_indices),
          'transitions':len(obs_rows),'state_dim':int(np.asarray(obs_rows).shape[-1]),
          'force_dim':int(np.asarray(force_rows).shape[-1]),'delta_q_dim':int(np.asarray(dq_rows).shape[-1]),
          'force_points':list(cmd.point_names)}), flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
