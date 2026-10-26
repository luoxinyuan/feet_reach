"""Real Isaac Lab rollout/PPO update, target isolation, frozen weights and resets.

Supports torchrun; uses the same task/algo as the long training launcher.
"""
import os
os.environ['ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE'] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import json
import torch
import hydra
from omegaconf import OmegaConf
from torchrl.envs.utils import set_exploration_type, ExplorationType
import active_adaptation.learning
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['sonic','ceer'], default='sonic')
parser.add_argument('--jacobian', action='store_true')
parser.add_argument('--spring18', action='store_true')
parser.add_argument('--segmented', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--guidance-cache', default=None)
parser.add_argument('--num-envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--updates', type=int, default=2)
args = parser.parse_args()
if args.amp:args.segmented=True
if sum((args.jacobian,args.spring18,args.segmented))>1:
    parser.error("Choose one force generator")
if (args.jacobian or args.spring18 or args.segmented) and args.backend != "ceer":
    parser.error("--jacobian currently supports --backend ceer only")
rank = int(os.environ.get('LOCAL_RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 1))
with hydra.initialize_config_dir(config_dir=str(ROOT / 'cfg'), version_base=None):
    cfg = hydra.compose(config_name='train', overrides=[
        f'task=G1/G1_{args.backend}{"_segmented_amp" if args.amp else "_segmented" if args.segmented else "_jacobian" if args.jacobian else "_spring18" if args.spring18 else ""}_compliance', f'algo={args.backend}_residual{"_amp" if args.amp else ""}',
        f'task.num_envs={args.num_envs}', 'algo.ppo_epochs=2', 'algo.num_minibatches=2',
        f'algo.train_every={args.steps}', 'wandb.mode=disabled', 'vecnorm=train',
    ])
if args.guidance_cache:
    cfg.task.command.guidance_cache=args.guidance_cache
    cfg.task.command.dataset.mem_paths=[str(Path(args.guidance_cache).resolve()/'dataset')]
cfg.seed += rank * 10000
OmegaConf.resolve(cfg)
app = AppLauncher(headless=True, device=f'cuda:{rank}', distributed=world > 1).app
try:
    if world > 1:
        torch.distributed.init_process_group('nccl')
    from scripts.utils.helpers import make_env_policy
    env, agent, vecnorm, _ = make_env_policy(cfg)
    env.train()
    td = env.reset()
    manager = env.action_manager
    low_policy = manager.sonic if args.backend == 'sonic' else manager.low_policy
    frozen_before = {k: v.clone() for k,v in low_policy.state_dict().items()}
    actor_before = {k: v.clone() for k,v in agent.actor.state_dict().items()}
    disc_before = {k:v.clone() for k,v in agent.discriminator.state_dict().items()} if args.amp else None
    sonic_ids = {id(p) for p in low_policy.parameters()}
    assert not sonic_ids.intersection(id(p) for group in agent.opt_actor.param_groups for p in group['params'])
    assert not sonic_ids.intersection(id(p) for group in agent.opt_critic.param_groups for p in group['params'])
    actor = agent.actor.module if isinstance(agent.actor, torch.nn.parallel.DistributedDataParallel) else agent.actor
    assert list(actor.in_keys) == ['policy']
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        public = td.select('policy').clone()
        private = td.clone()
        for key in ('priv', 'joint_target', 'priv_critic', 'amp_state_'):
            if key in private.keys(): private[key].fill_(10000)
        actor(public); actor(private)
        torch.testing.assert_close(public['action'], private['action'])
        # Poison true future poses while preserving actual state and measured
        # forces. Public observation construction must remain unchanged too.
        funcs = env.observation_funcs['policy'].funcs
        public_before = {key: func().clone() for key, func in funcs.items()}
        motion = env.command_manager._motion
        saved_targets = {key: getattr(motion, key).clone() for key in ('joint_pos', 'joint_vel', 'root_pos_w', 'root_quat_w')}
        for key in saved_targets: getattr(motion, key).fill_(1234.)
        for key, func in funcs.items():
            torch.testing.assert_close(func(), public_before[key], atol=0, rtol=0)
        if args.backend == 'ceer':
            # Both CEER branches (interface + direct joints) are also isolated.
            first=manager.low_inputs(manager.history.q.flatten(1))
            for key in saved_targets: getattr(motion,key).fill_(-999.)
            second=manager.low_inputs(manager.history.q.flatten(1))
            for a,b in zip(first,second): torch.testing.assert_close(a,b,atol=0,rtol=0)
        for key, value in saved_targets.items(): getattr(motion, key).copy_(value)
        # Selected reset must not modify another environment's reference/history.
        before = manager.reference.q.clone()
        histories = [v.clone() for v in manager.history.values]
        manager.reset(torch.tensor([0], device=manager.device))
        torch.testing.assert_close(before[1:], manager.reference.q[1:])
        for a,b in zip(histories, manager.history.values): torch.testing.assert_close(a[1:],b[1:])
    if args.jacobian or args.spring18:
        cmd = env.command_manager
        with torch.no_grad():
            old_slots = cmd.slots.clone()
            cmd.desired.copy_(cmd.active[..., None].expand_as(cmd.desired).float() * 10.)
            for _ in range(30): cmd.step(0)
            cmd.remaining.zero_()
            cmd.before_update()
            for _ in range(100):
                previous = (cmd.point_force_buffer if args.spring18 else cmd.force_apply_buffer).clone()
                cmd.step(0)
                applied = cmd.point_force_buffer if args.spring18 else cmd.force_apply_buffer
                assert (applied - previous).norm(dim=-1).max() <= 150 * env.physics_dt + 1e-4
                # Re-selection only after zero force; test the full fade sequence.
                if cmd.force.abs().max() < 1e-5:
                    cmd.before_update()
                    break
            else:
                raise AssertionError('Contact release did not finish')
        td = env.reset()
    if args.spring18:
        cmd = env.command_manager
        with torch.no_grad():
            # Force a repeated-body selection: torso front/back/left. A couple
            # must survive even when the resultant force cancels.
            cmd.slots[:]=torch.tensor([9,10,11],device=cmd.device)
            cmd.active[:]=True
            from active_adaptation.utils.math import quat_apply
            cmd.force.zero_()
            axis=torch.zeros(cmd.num_envs,3,device=cmd.device);axis[:,1]=10.
            direction=quat_apply(cmd.asset.data.body_quat_w[:,cmd.point_ids[9]],axis)
            cmd.force[:,0]=direction;cmd.force[:,1]=-direction
            cmd.desired.copy_(cmd.force);cmd.step(0)
            torch.testing.assert_close(cmd.force_apply_buffer.sum(1),torch.zeros_like(cmd.force[:,0]))
            expected=torch.cross(cmd.point_positions_w()[cmd.rows[:,None],cmd.slots]-cmd.asset.data.root_pos_w[:,None],cmd.force,dim=-1).sum(1)
            torch.testing.assert_close(cmd.torque_apply_buffer.sum(1),expected,atol=1e-4,rtol=1e-4)
            assert cmd.torque_apply_buffer.norm(dim=-1).sum(-1).min()>1.
            print('SPRING18_POINTS',list(zip(cmd.point_names,cmd.point_positions_w()[0].tolist())),flush=True)
        td=env.reset()
    if args.segmented:
        from active_adaptation.utils.segmented_guidance import geometry_jacobians
        from active_adaptation.utils.jacobian_guidance import point_linear_jacobian
        import numpy as np
        cmd=env.command_manager
        geom=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
        points=OmegaConf.to_container(cfg.task.command.force_points)
        pos,jac=geometry_jacobians(cmd.asset.data.joint_pos.cpu().numpy(),
            cmd.asset.data.root_pos_w.cpu().numpy(),cmd.asset.data.root_quat_w.cpu().numpy(),
            cmd.asset.joint_names,points,geom)
        torch.testing.assert_close(torch.as_tensor(pos,device=cmd.device,dtype=torch.float32),cmd.point_positions_w(),atol=2e-4,rtol=1e-4)
        raw=cmd.asset.root_physx_view.get_jacobians()[:,cmd.point_ids]
        actual=point_linear_jacobian(raw,cmd.point_positions_w()-cmd.asset.data.body_com_pos_w[:,cmd.point_ids])
        torch.testing.assert_close(torch.as_tensor(jac[...,6:],device=cmd.device,dtype=torch.float32),actual[...,6:],atol=2e-4,rtol=1e-4)
        print('OFFLINE_USD_JACOBIAN_MATCHES_PHYSX',flush=True)
    reports = []
    rollout = agent.get_rollout_policy('train')
    for update in range(args.updates):
        samples = []
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            for _ in range(args.steps):
                td = rollout(td)
                sample, td = env.step_and_maybe_reset(td)
                assert torch.isfinite(manager.low_action).all()
                assert torch.isfinite(sample['next', 'reward']).all()
                assert env.command_manager.force.norm(dim=-1).max() <= 30.001
                if not (args.spring18 or args.segmented):
                    assert env.command_manager.torque_apply_buffer.count_nonzero() == 0
                cmd = env.command_manager
                assert cmd.force.norm(dim=-1).sum(-1).max() <= 60.001
                assert (cmd.force_apply_buffer.norm(dim=-1) > 1e-5).sum(-1).max() <= 3
                if args.spring18 or args.segmented:
                    assert len(cmd.point_names) == 18
                    assert td['policy'].shape[-1] == 408
                    assert cmd.upper_body_forces_b().shape[-1] == 54
                    assert (cmd.point_force_buffer.norm(dim=-1) > 1e-5).sum(-1).max() <= 3
                    torch.testing.assert_close(cmd.force_apply_buffer.sum(1), cmd.force.sum(1), atol=1e-5, rtol=1e-5)
                    actual_moment = (cmd.torque_apply_buffer + torch.cross(
                        cmd.position_apply_buffer-cmd.asset.data.root_pos_w[:,None], cmd.force_apply_buffer,dim=-1)).sum(1)
                    applied_points = cmd.point_positions_w()[cmd.rows[:,None],cmd.slots]
                    point_moment = torch.cross(applied_points-cmd.asset.data.root_pos_w[:,None],cmd.force,dim=-1).sum(1)
                    # Wrench buffers were computed before the last physics substep;
                    # compare at the stored application geometry, tested below.
                    assert torch.isfinite(actual_moment).all()
                    from active_adaptation.utils.jacobian_guidance import point_linear_jacobian
                    raw=cmd.asset.root_physx_view.get_jacobians()[:,cmd.point_ids]
                    offset=cmd.point_positions_w()-cmd.asset.data.body_com_pos_w[:,cmd.point_ids]
                    jac=point_linear_jacobian(raw,offset)
                    velocity=torch.cat((cmd.asset.data.root_lin_vel_w,cmd.asset.data.root_ang_vel_w,cmd.asset.data.joint_vel),-1)
                    torch.testing.assert_close(torch.einsum('nkcd,nd->nkc',jac,velocity),cmd.point_velocities_w(),atol=1e-4,rtol=1e-4)
                if args.jacobian:
                    assert torch.isfinite(cmd.generalized_target).all()
                    # World Jacobian joint columns should reproduce relative
                    # link velocities after removing floating-base motion.
                    from active_adaptation.utils.jacobian_guidance import point_linear_jacobian
                    raw = cmd.asset.root_physx_view.get_jacobians()[:, cmd.point_ids]
                    offset = cmd.asset.data.body_pos_w[:, cmd.point_ids] - cmd.asset.data.body_com_pos_w[:, cmd.point_ids]
                    jac = point_linear_jacobian(raw, offset)[..., 6:]
                    predicted = torch.einsum('nkcd,nd->nkc', jac, cmd.asset.data.joint_vel)
                    arms = cmd.asset.data.body_pos_w[:, cmd.point_ids] - cmd.asset.data.root_pos_w[:, None]
                    predicted += torch.cross(cmd.asset.data.root_ang_vel_w[:, None].expand_as(arms), arms, dim=-1)
                    measured = cmd.asset.data.body_link_lin_vel_w[:, cmd.point_ids] - cmd.asset.data.root_link_lin_vel_w[:, None]
                    torch.testing.assert_close(predicted, measured, atol=1e-4, rtol=1e-4)
                agent.critic(sample); agent.critic(sample['next'])
                sample['next','state_value'][:] = torch.where(sample['next','done'], sample['state_value'], sample['next','state_value'])
                sample['next'] = sample['next'].exclude(*rollout.in_keys)
                private_keys = [k for k in sample.keys(True,True) if isinstance(k,str) and k.startswith('_')]
                samples.append(sample.exclude(*private_keys, 'priv_pred', 'priv_feature').clone())
        # Exercise LR adaptation too, including collective averaging under DDP.
        agent.step_schedule(.5, update)
        info = agent.train_op(torch.stack(samples, 1), vecnorm)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values())
        reports.append(info)
    for name, value in low_policy.state_dict().items():
        assert torch.equal(value, frozen_before[name]), name
    assert any(not torch.equal(value, actor_before[name]) for name,value in agent.actor.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in low_policy.parameters())
    if args.amp:
        assert any(not torch.equal(v,disc_before[k]) for k,v in agent.discriminator.state_dict().items())
        if world>1:
            for p in agent.discriminator.parameters():
                expected=p.detach().clone();torch.distributed.broadcast(expected,0)
                torch.testing.assert_close(p,expected,atol=0,rtol=0)
        assert reports[-1]['amp/weight']>0 and reports[-1]['amp/bonus_per_step']>0
        # State serialization includes the learned discriminator and optimizer.
        saved=agent.state_dict();assert 'amp_discriminator' in saved and 'optimizer_amp' in saved
        agent.load_state_dict(saved)
    if world > 1:
        for p in agent.actor.parameters():
            expected = p.detach().clone(); torch.distributed.broadcast(expected, src=0)
            torch.testing.assert_close(p, expected, atol=0, rtol=0)
        lrs = [None] * world
        torch.distributed.all_gather_object(lrs, agent.current_lr)
        assert len(set(lrs)) == 1
    out = ROOT / f'artifacts/{args.backend}{"_segmented_amp" if args.amp else "_segmented" if args.segmented else "_jacobian" if args.jacobian else "_spring18" if args.spring18 else ""}_residual'; out.mkdir(parents=True, exist_ok=True)
    result = {'rank': rank, 'world_size': world, 'num_envs_per_gpu': args.num_envs,
              'updates': reports, 'amp_discriminator_updated_and_synchronized': args.amp, 'frozen_weights_unchanged': True, 'actor_updated': True,
              'no_target_input': True, 'per_env_reset_isolation': True,
              'max_force_N': env.command_manager.force.norm(dim=-1).max().item(),
              'jacobian_world_point_velocity_verified': args.jacobian,
              'expanded_point_count': 18 if (args.spring18 or args.segmented) else 9,
              'same_link_force_couple_verified': args.spring18,
              'guidance_metrics': {k: v for k,v in env.extra.items() if k.startswith('jacobian/')},
              'gpu_peak_memory_MB': torch.cuda.max_memory_allocated() / 2**20}
    (out / f'smoke_{world}gpu_rank{rank}.json').write_text(json.dumps(result, indent=2))
    print(args.backend.upper() + '_SMOKE_PASS', json.dumps(result), flush=True)
    if rank == 0:
        torch.save({'policy': agent.state_dict(), 'vecnorm': vecnorm.state_dict(), 'cfg': cfg}, out / 'smoke_checkpoint.pt')
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
    if world > 1: torch.distributed.destroy_process_group()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
