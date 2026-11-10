"""Roll out a trained 3-point CEER force teacher and collect (force, realized delta-q)."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, json, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--motion-count', type=int, default=160)
parser.add_argument('--steps-per-motion', type=int, default=100)
parser.add_argument('--seed', type=int, default=20261103)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

checkpoint = args.checkpoint.resolve()
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
OmegaConf.set_struct(cfg, False); OmegaConf.set_readonly(cfg, False)
cfg.checkpoint_path = str(checkpoint)
cfg.vecnorm = 'eval'; cfg.seed = args.seed; cfg.task.num_envs = 1
cfg.task.max_episode_length = 1200
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
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager; robot = cmd.asset; dataset = cmd.dataset
    if cfg.algo.phase != 'teacher':
        raise RuntimeError(f"Expected a GT-force teacher checkpoint, got phase={cfg.algo.phase}")
    policy = agent.get_rollout_policy('deploy')
    if list(policy.in_keys) != ['force_gt_', 'policy']:
        raise RuntimeError(f'Unexpected teacher actor inputs: {policy.in_keys}')
    if len(dataset.datasets) != 2 or len(dataset.probs) != 2:
        raise RuntimeError('Expected root3 + crouch datasets with the established 1.0:0.05 mixture')
    if len(cmd.point_ids) != 3:
        raise RuntimeError(f'Expected 3 force points, got {len(cmd.point_ids)}')

    rng = np.random.default_rng(args.seed)
    # Sample distinct clips according to the same dataset-level path weights used in training.
    clip_ds = np.concatenate([np.full(ds.num_motions, i, dtype=np.int32)
                              for i, ds in enumerate(dataset.datasets)])
    clip_mid = np.concatenate([np.arange(ds.num_motions, dtype=np.int32)
                               for ds in dataset.datasets])
    probs = np.concatenate([np.full(ds.num_motions, float(dataset.probs[i].item()) / ds.num_motions,
                                   dtype=np.float64)
                           for i, ds in enumerate(dataset.datasets)])
    probs /= probs.sum()
    selected = rng.choice(len(clip_ds), size=min(args.motion_count, len(clip_ds)),
                          replace=False, p=probs)
    manager = env.action_manager
    joint_ids = torch.as_tensor(manager.joint_ids, device=env.device)
    joint_names = [robot.joint_names[int(i)] for i in joint_ids.cpu().tolist()]
    force_rows, delta_rows, clip_rows, step_rows, source_rows = [], [], [], [], []
    cmd.zero_init_prob = 1.0
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for n, global_id in enumerate(selected):
            source_id, mid = int(clip_ds[global_id]), int(clip_mid[global_id])
            dataset.fix_ds = source_id; dataset.fix_motion_id = mid
            dataset._populate_buffer_full(target='A')
            td = env.reset()
            actual_id = int(cmd._motion.motion_id[0, 0])
            length = int(cmd.lengths[0].item())
            steps = min(args.steps_per_motion, length - 1)
            if steps <= 0:
                continue
            for step in range(steps):
                q0 = robot.data.joint_pos[0, joint_ids].detach().clone()
                # Actor input is normalized by 30 N and expressed in root coordinates.
                force_N = td['force_gt_'][0].reshape(3, 3).detach().float().cpu().numpy() * 30.0
                action_td = policy(td)
                next_td = env.step(action_td)
                q1 = robot.data.joint_pos[0, joint_ids].detach().clone()
                force_rows.append(force_N.reshape(-1))
                delta_rows.append((q1 - q0).float().cpu().numpy())
                clip_rows.append(actual_id); step_rows.append(step); source_rows.append(source_id)
                td = next_td['next']
            if (n + 1) % 20 == 0:
                print(f'collected {n+1}/{len(selected)} clips; {len(force_rows)} transitions', flush=True)

    np.savez_compressed(args.output, force_N=np.asarray(force_rows, dtype=np.float32),
        delta_q_rad=np.asarray(delta_rows, dtype=np.float32), clip_id=np.asarray(clip_rows, dtype=np.int32),
        dataset_id=np.asarray(source_rows, dtype=np.int8), step=np.asarray(step_rows, dtype=np.int16),
        point_names=np.asarray(cmd.point_names), joint_names=np.asarray(joint_names),
        seed=np.asarray(args.seed), motion_count=np.asarray(len(selected)),
        steps_per_motion=np.asarray(args.steps_per_motion), checkpoint=np.asarray(str(checkpoint)))
    summary = {'output': str(args.output.resolve()), 'clips_sampled': int(len(selected)),
        'transitions': int(len(force_rows)), 'force_dim': 9, 'joint_delta_dim': len(joint_names),
        'points': list(cmd.point_names), 'joint_names': joint_names,
        'dataset_mix': [float(x) for x in dataset.probs.cpu().tolist()],
        'seed': args.seed, 'force_units': 'N, root frame', 'target': 'realized next-step joint_pos delta, radians'}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('ROLLOUT_COLLECTION_COMPLETE', json.dumps(summary), flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
