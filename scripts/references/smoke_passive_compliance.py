"""Exercise physics forces, target isolation, PPO gradients and phase handoff."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--phase',default='train');p.add_argument('--checkpoint')
p.add_argument('--task',default='G1/G1_passive_compliance')
p.add_argument('--output',type=Path,default=ROOT/'artifacts/passive_compliance');args=p.parse_args()
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=[f'task={args.task}','algo=passive_compliance',f'algo.phase={args.phase}','task.num_envs=8','algo.ppo_epochs=1','algo.num_minibatches=1','algo.train_every=16','wandb.mode=disabled'])
if args.checkpoint:cfg.checkpoint_path=args.checkpoint
OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device='cuda:0').app
try:
 from scripts.utils.helpers import make_env_policy
 env,policy,vecnorm,_=make_env_policy(cfg);env.train();carry=env.reset();cmd=env.command_manager
 assert len(cmd.joint_idx_asset)==29
 assert set(cfg.task.observation.policy)=={'root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','ee_external_force_b'}
 # Deployed actor must execute with policy observations alone, even if all
 # privileged labels are absent. Changing privileged data cannot change loc.
 student=policy.get_rollout_policy('deploy')
 with torch.no_grad():
  public=carry.select('policy').clone();student(public)
  private=carry.clone();private['joint_target'].fill_(1000);private['priv'].fill_(-1000);student(private)
  torch.testing.assert_close(public['loc'],private['loc'])
 assert list(student.in_keys)==['policy'],student.in_keys
 rollout=policy.get_rollout_policy('train');steps=[];max_force=0.
 with torch.no_grad():
  for _ in range(cfg.algo.train_every):
   carry=rollout(carry);td,carry=env.step_and_maybe_reset(carry)
   assert torch.isfinite(cmd.ee_force).all()
   assert cmd.ee_force.norm(dim=-1).max()<=cmd.force_limit+.001
   other=cmd.force_apply_buffer.clone();other[:,cmd.hand_ids]=0.;assert other.count_nonzero()==0
   max_force=max(max_force,cmd.ee_force.norm(dim=-1).max().item())
   policy.critic(td);policy.critic(td['next'])
   td['next','state_value'][:]=torch.where(td['next','done'],td['state_value'],td['next','state_value'])
   td['next']=td['next'].exclude(*rollout.in_keys)
   private_keys=[k for k in td.keys(True,True) if isinstance(k,str) and k.startswith('_')]
   steps.append(td.exclude(*private_keys,'priv_pred','priv_feature').clone())
 assert max_force>0
 batch=torch.stack(steps,dim=1);policy.step_schedule(.5,1);info=policy.train_op(batch,vecnorm)
 assert info and all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values()),info
 # Zero external force is exact on hold tasks, regardless of pose error.
 cmd.active_hands.zero_();cmd.step(0);assert cmd.force_apply_buffer.count_nonzero()==0
 # A timeout advances the transition without resetting the physical robot.
 cmd.age.fill_(cmd.transition_timeout);before=cmd.asset.data.joint_pos.clone();count=cmd.transition_count
 cmd.before_update();assert cmd.transition_count==count+env.num_envs
 torch.testing.assert_close(before,cmd.asset.data.joint_pos)
 output=args.output;output.mkdir(parents=True,exist_ok=True)
 report=dict(task=args.task,phase=args.phase,force_limit_N=cmd.force_limit,max_force_N=max_force,student_in_keys=list(student.in_keys),metrics={k:float(v) for k,v in info.items()})
 (output/f'smoke_{args.phase}.json').write_text(json.dumps(report,indent=2))
 torch.save({'policy':policy.state_dict(),'vecnorm':vecnorm.state_dict(),'cfg':cfg},output/f'smoke_{args.phase}.pt')
 print('PASSIVE_COMPLIANCE_SMOKE_OK',report,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True
 env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
