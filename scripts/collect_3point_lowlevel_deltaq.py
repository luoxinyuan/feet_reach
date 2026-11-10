"""Collect force -> realized delta-q data by rolling out the force-finetuned CEER teacher."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import argparse, json, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--total-transitions', type=int, default=4_000_000_000)
    p.add_argument('--num-envs', type=int, default=16_384)
    p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--expected-crouch-weight', type=float, default=0.05)
    a = p.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_count = a.total_transitions // world + (rank < a.total_transitions % world)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher

    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
    OmegaConf.set_struct(cfg, False); OmegaConf.set_readonly(cfg, False)
    cfg.checkpoint_path = str(a.checkpoint.resolve())
    cfg.vecnorm = 'eval'; cfg.seed = a.seed + rank
    cfg.task.num_envs = a.num_envs
    cfg.task.observation.force_gt_ = {'dense_force_gt': {}}
    cfg.wandb.mode = 'disabled'; cfg.eval_render = False; cfg.headless = True
    del state

    app = AppLauncher(headless=True, device=f'cuda:{rank}', enable_cameras=False,
        kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/omni.tbb.globalcontrol/maxThreadCount=16').app
    try:
        # make_env_policy broadcasts the loaded checkpoint and observation
        # normalizer from rank 0, matching the training launcher semantics.
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group('nccl')
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from scripts.utils.helpers import make_env_policy
        torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
        env, agent, _, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval()
        cmd = env.command_manager; robot = cmd.asset; manager = env.action_manager
        if cfg.algo.phase != 'train':
            raise RuntimeError(
                f'Expected a teacher-phase CEER checkpoint, got algo.phase={cfg.algo.phase!r}. '
                'The rollout must use actor_teacher, not the finetuned student.'
            )
        if len(cmd.point_ids) != 3 or list(cmd.point_names) != ['pelvis', 'left_hand_mimic', 'right_hand_mimic']:
            raise RuntimeError(f'Unexpected physical force points: {getattr(cmd, "point_names", None)}')
        mix = [float(x) for x in cmd.dataset.probs.cpu().tolist()]
        if len(mix) != 2 or abs(mix[1] - a.expected_crouch_weight / (1.0 + a.expected_crouch_weight)) > 1e-5:
            raise RuntimeError(f'Dataset mixture does not preserve 1.0:0.05 weights: {mix}')
        if not hasattr(cmd, 'dense_force_gt'):
            raise RuntimeError('Force observation dense_force_gt is unavailable')
        if len(cmd.all_joint_idx_asset) != 29:
            raise RuntimeError(f'Expected 29 CEER joints, got {len(cmd.all_joint_idx_asset)}')

        prefix = a.output_dir / f'rank{rank:02d}'
        force_mm = np.lib.format.open_memmap(str(prefix.with_name(prefix.name + '_force_N.npy')),
            mode='w+', dtype=np.float16, shape=(local_count, 9))
        delta_mm = np.lib.format.open_memmap(str(prefix.with_name(prefix.name + '_delta_q_rad.npy')),
            mode='w+', dtype=np.float16, shape=(local_count, 29))
        clip_mm = np.lib.format.open_memmap(str(prefix.with_name(prefix.name + '_motion_id.npy')),
            mode='w+', dtype=np.int32, shape=(local_count,))

        joint_ids = torch.as_tensor(manager.joint_ids, device=env.device)
        if len(joint_ids) != 29:
            raise RuntimeError(f'Expected 29 policy joints, got {len(joint_ids)}')
        policy = agent.get_rollout_policy('teacher')
        print('Rollout actor: CEER actor_teacher with full-body joint-target input', flush=True)
        td = env.reset(); written = 0; steps = 0
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            while written < local_count:
                q0 = robot.data.joint_pos[:, joint_ids].detach().clone()
                force_n = td['force_gt_'].reshape(a.num_envs, 9).float() * 30.0
                motion_id = cmd._motion.motion_id[:, 0].to(torch.int32)
                action_td = policy(td)
                stepped = env.step(action_td)
                q1 = robot.data.joint_pos[:, joint_ids].detach()
                count = min(a.num_envs, local_count - written)
                sl = slice(written, written + count)
                force_mm[sl] = force_n[:count].cpu().numpy().astype(np.float16)
                delta_mm[sl] = (q1[:count] - q0[:count]).cpu().numpy().astype(np.float16)
                clip_mm[sl] = motion_id[:count].cpu().numpy()
                written += count; steps += 1
                # Detach the next observation from the just-finished transition.
                # Keeping TensorDict views alive here can retain the whole chain
                # of prior `next` values and grow CUDA memory throughout rollout.
                next_td = stepped['next'].clone()
                del action_td, stepped, q0, force_n, motion_id, q1
                td = next_td
                del next_td
                if steps % 100 == 0:
                    force_mm.flush(); delta_mm.flush(); clip_mm.flush()
                    if steps % 1000 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f'rank={rank} transitions={written:,}/{local_count:,} env_steps={steps:,} cuda_allocated_GB={allocated:.2f} cuda_reserved_GB={reserved:.2f}', flush=True)
        force_mm.flush(); delta_mm.flush(); clip_mm.flush()
        manifest = {'rank': rank, 'world_size': world, 'transitions': local_count,
            'env_steps': steps, 'num_envs': a.num_envs, 'seed': a.seed + rank,
            'force_points': list(cmd.point_names), 'force_frame': 'root frame', 'force_unit': 'N',
            'joint_names': list(robot.joint_names[int(i)] for i in joint_ids.cpu().tolist()),
            'label': 'measured next-step articulation joint-position delta, radians',
            'dataset_path_weights': [1.0, a.expected_crouch_weight], 'normalized_dataset_probabilities': mix,
            'checkpoint': str(a.checkpoint.resolve())}
        prefix.with_name(prefix.name + '_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('RANK_COLLECTION_COMPLETE ' + json.dumps(manifest), flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    else:
        app.close()

if __name__ == '__main__': main()
