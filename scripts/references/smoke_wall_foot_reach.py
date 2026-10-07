"""Finite Isaac/PPO integration check; does not assess a trained controller."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)
import argparse
p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['train','adapt','finetune'],default='train')
p.add_argument('--checkpoint',default=None)
p.add_argument('--foot-force',action='store_true')
args=p.parse_args()
import torch
import hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
    cfg=hydra.compose(config_name='train',overrides=['task=G1/G1_wall_foot_reach',f'+exp={args.phase}',
        'task.num_envs=4','algo.symmetry_augmentation=false','algo.ppo_epochs=1','algo.num_minibatches=1',
        'algo.train_every=16','wandb.mode=disabled'])
if args.foot_force:
    for key in ('ramp_up_range','hold_range','ramp_down_range','rest_range'):
        cfg.task.command.foot_force[key]=[2,2]
    cfg.task.command.foot_force.zero_prob=0.
if args.checkpoint: cfg.checkpoint_path=args.checkpoint
OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device='cuda:0').app
try:
    from isaaclab.sim import SimulationContext
    from scripts.utils.helpers import make_env_policy
    env,policy,vecnorm,_=make_env_policy(cfg)
    env.train()
    carry=env.reset()
    cmd=env.command_manager
    assert carry['joint_target'].shape[-1]==29
    from active_adaptation.utils.math import quat_apply_inverse, quat_apply
    expected=quat_apply_inverse(cmd.asset.data.root_quat_w,cmd.target_sole_world()-cmd.asset.data.root_pos_w)
    torch.testing.assert_close(cmd.target_foot_pos_b(),expected)
    actual=cmd.asset.data.body_pos_w[:,cmd.left_asset]+quat_apply(cmd.asset.data.body_quat_w[:,cmd.left_asset],cmd.sole_offset.expand(4,-1))
    initial_error=torch.linalg.vector_norm(actual-cmd.target_sole_world(),dim=-1).max().item()
    assert initial_error < .002,initial_error
    assert cmd._motion.joint_pos.dtype==torch.float32
    xyz=torch.tensor([.1,.05,-.4],device=cmd.device)
    cmd.set_target_foot_pos_b(xyz)
    torch.testing.assert_close(cmd.target_foot_pos_b(),xyz.expand(4,-1),atol=1e-6,rtol=1e-5)
    cmd.set_target_foot_pos_b(None)
    if args.foot_force:
        force_peak=0.
        for _ in range(12):
            cmd.step(0)
            force=cmd.force_apply_buffer.clone()
            force_peak=max(force_peak,float(force.norm(dim=-1).max()))
            assert float(force.norm(dim=-1).max())<=20.00001
            other=force.clone();other[:,cmd.left_asset]=0
            assert other.count_nonzero()==0 and cmd.torque_apply_buffer.count_nonzero()==0
            expected=cmd.asset.data.body_pos_w[:,cmd.left_asset]+quat_apply(cmd.asset.data.body_quat_w[:,cmd.left_asset],cmd.sole_offset.expand(4,-1))
            torch.testing.assert_close(cmd.position_apply_buffer[:,cmd.left_asset],expected)
            cmd.step(1)
            torch.testing.assert_close(cmd.force_apply_buffer,force)
        assert force_peak>0
        cmd.env.eval();cmd.step(0)
        assert not cmd.force_apply_world and cmd.force_apply_buffer.count_nonzero()==0
        cmd.env.train();env.train();carry=env.reset()
        print('FOOT_FORCE_APPLICATION_OK',force_peak,flush=True)
    rollout=policy.get_rollout_policy('train')
    steps=[]
    with torch.no_grad():
        for _ in range(cfg.algo.train_every):
            carry=rollout(carry)
            td,carry=env.step_and_maybe_reset(carry)
            policy.critic(td);policy.critic(td['next'])
            td['next','state_value'][:]=torch.where(td['next','done'],td['state_value'],td['next','state_value'])
            td['next']=td['next'].exclude(*rollout.in_keys)
            private=[k for k in td.keys(True,True) if isinstance(k,str) and k.startswith('_')]
            steps.append(td.exclude(*private,'priv_pred','priv_feature').clone())
    batch=torch.stack(steps,dim=1)
    policy.step_schedule(.5,1)
    info=policy.train_op(batch,vecnorm)
    assert info and all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values()),info
    # Student must run with no teacher labels or privileged observations.
    student=policy.get_rollout_policy('eval')
    only_policy=carry.select('policy','is_init',strict=False).clone()
    if args.phase!='train':
        student(only_policy)
        assert only_policy['action'].shape==(4,29)
    import json
    report={'phase':args.phase,'policy_dim':carry['policy'].shape[-1],
            'joint_target_dim':29,'initial_sole_error_m':initial_error,
            'metrics':{k:float(v) for k,v in info.items()}}
    output=ROOT/'artifacts/foot_reach'
    if args.foot_force: output=output/'force_smoke'
    output.mkdir(parents=True,exist_ok=True)
    (output/f'smoke_{args.phase}.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save({'policy':policy.state_dict(),'vecnorm':vecnorm.state_dict()},output/f'smoke_{args.phase}.pt')
    print('FOOT_REACH_SMOKE_OK',args.phase,'policy_dim',carry['policy'].shape[-1],
          'joint_target_dim',carry['joint_target'].shape[-1],'metrics',info,flush=True)
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(1)
else:
    app.close()
