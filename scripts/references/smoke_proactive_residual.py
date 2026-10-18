"""Exercise frozen CEER, nominal-target isolation, and both residual PPO phases."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys, argparse, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--phase',default='train');p.add_argument('--checkpoint');p.add_argument('--task',default='G1_proactive_residual');p.add_argument('--output',type=Path,default=ROOT/'artifacts/proactive_residual');args=p.parse_args()
import torch, hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=[f'task=G1/{args.task}','algo=proactive_residual',f'algo.phase={args.phase}','task.num_envs=8','algo.ppo_epochs=1','algo.num_minibatches=1','algo.train_every=8','vecnorm=train','wandb.mode=disabled',f'task.action.low_policy.checkpoint_path={ROOT}/.cache/ceer/checkpoint_final.pt'])
if args.checkpoint: cfg.checkpoint_path=args.checkpoint
if args.phase=='adapt':
 cfg.vecnorm='eval'
 if 'gentle' in cfg.task.command:cfg.task.command.gentle.curriculum_enabled=False
OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device='cuda:0').app
try:
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.learning.modules.pointnet import GeometryPointNet
 env,policy,vecnorm,_=make_env_policy(cfg);env.train();carry=env.reset();cmd=env.command_manager;act=env.action_manager
 assert act.action_dim==3
 assert not any(p.requires_grad for p in act.low_policy.policy.parameters())
 frozen={k:p.detach().clone() for k,p in act.low_policy.policy.named_parameters()}
 ck=torch.load(ROOT/'.cache/ceer/checkpoint_final.pt',map_location='cpu',weights_only=False)
 assert list(cfg.task.observation.policy)==list(ck['cfg'].task.observation.policy)
 for name,module in act.low_policy.policy.named_children():
  if any(True for _ in module.parameters()):
   for key,value in module.state_dict().items():
    torch.testing.assert_close(value.cpu(),ck['policy'][name][key].cpu(),rtol=0,atol=0)
 assert env.observation_spec['policy'].shape[-1]==ck['vecnorm']['_extra_state']['policy_sum'].shape[-1]
 funcs=env.reward_groups['tracking'].funcs
 if 'gentle' in cfg.task.command:
  enabled={k for k,v in funcs.items() if v.enabled}
  expected={'gentle_tracking','gentle_requirement','gentle_approach','gentle_success','gentle_residual_smooth','gentle_impulse'}
  if args.task.endswith('_B'):expected|={'gentle_growth','gentle_peak'}
  assert enabled==expected,enabled
 else:
  assert 'left_ee_tracking' in funcs and 'contact_requirement' in funcs

 nominal=cmd.target_ee_pos_b().clone();reference=act._get_ee_command_reference().clone();before=(cmd.gentle_tracking() if 'gentle' in cfg.task.command else cmd.left_ee_tracking()).clone()
 decoded=act._decode_ee_command(torch.ones(8,3,device=cmd.device))
 torch.testing.assert_close(decoded[:,3:],reference[:,3:])
 torch.testing.assert_close(decoded[:,:3]-reference[:,:3],torch.full_like(nominal,.15*float(torch.tanh(torch.tensor(1.)))))
 cmd.set_root_and_wrist_6d_command(decoded)
 torch.testing.assert_close(cmd.target_ee_pos_b(),nominal);torch.testing.assert_close(cmd.gentle_tracking() if 'gentle' in cfg.task.command else cmd.left_ee_tracking(),before)
 if 'gentle' in cfg.task.command:
  from active_adaptation.utils.math import quat_apply_inverse
  # Test task gates against an independent physical state before any PPO update.
  actual=cmd.asset.data.body_pos_w[:,cmd.hand_asset].clone()
  goal=actual.clone();goal[:,0]+=.1
  cmd.set_target_ee_pos_b(quat_apply_inverse(cmd.asset.data.root_quat_w,goal-cmd.asset.data.root_pos_w))
  cmd.wall_present.fill_(False)
  torch.testing.assert_close(cmd.gentle_tracking(),torch.full((8,1),float(torch.exp(torch.tensor(-.5))),device=cmd.device),atol=1e-5,rtol=1e-5)
  cmd.wall_present.fill_(True);cmd.wall_x.copy_(actual[:,0]-env.scene.env_origins[:,0]+.05)
  torch.testing.assert_close(cmd.gentle_tracking(),torch.ones(8,1,device=cmd.device),atol=1e-5,rtol=1e-5)
  velocity=cmd.asset.data.body_lin_vel_w[:,cmd.hand_asset].clone()
  cmd.asset.data.body_lin_vel_w[:,cmd.hand_asset,0]=.5
  cmd.wall_contact_this_step.fill_(False)
  cmd.step_schedule(.2)
  torch.testing.assert_close(cmd.gentle_approach(),torch.full((8,1),-.25,device=cmd.device),atol=1e-5,rtol=1e-5)
  cmd.step_schedule(0.)
  if cmd.curriculum_enabled:assert not cmd.gentle_approach().any()
  cmd.asset.data.body_lin_vel_w[:,cmd.hand_asset]=velocity
  cmd.wall_contact_this_step.fill_(True);cmd.contact_all_step.fill_(True)
  total_success=0.
  for _ in range(25):total_success+=cmd.gentle_success().sum().item()*env.step_dt
  assert abs(total_success-8*.2)<1e-5,total_success
  # Early contact in the 2 cm target band is allowed, more distant contact is not.
  cmd.wall_x.copy_(goal[:,0]-env.scene.env_origins[:,0]+.01)
  assert not cmd.gentle_requirement().any()
  cmd.wall_x.add_(.04)
  torch.testing.assert_close(cmd.gentle_requirement(),torch.full((8,1),-3.,device=cmd.device))
  cmd.set_target_ee_pos_b(None)
  carry=env.reset();cmd.step_schedule(.2)
 rollout=policy.get_rollout_policy('train');steps=[]
 with torch.no_grad():
  for _ in range(cfg.algo.train_every):
   carry=rollout(carry);td,carry=env.step_and_maybe_reset(carry)
   policy.critic(td);policy.critic(td['next'])
   td['next','state_value'][:]=torch.where(td['next','done'],td['state_value'],td['next','state_value'])
   td['next']=td['next'].exclude(*rollout.in_keys)
   private=[k for k in td.keys(True,True) if isinstance(k,str) and k.startswith('_')]
   steps.append(td.exclude(*private,'priv_pred','priv_feature').clone())
 batch=torch.stack(steps,dim=1);policy.step_schedule(.5,1);info=policy.train_op(batch,vecnorm)
 assert info and all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values()),info
 for k,v in act.low_policy.policy.named_parameters():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
 if args.phase=='train':
  grads=[p.grad.abs().sum().item() for m in policy.actor_teacher.modules() if isinstance(m,GeometryPointNet) for p in m.parameters() if p.grad is not None]
  assert sum(grads)>0
 public=carry.select('hl_policy','geometry_','is_init',strict=False).clone()
 policy.get_rollout_policy('eval')(public);assert public['action'].shape==(8,3)
 out=args.output;out.mkdir(exist_ok=True,parents=True)
 torch.save({'policy':policy.state_dict(),'vecnorm':vecnorm.state_dict(),'cfg':cfg},out/f'smoke_{args.phase}.pt')
 report={'phase':args.phase,'frozen_ceer_unchanged':True,'metrics':{k:float(v) for k,v in info.items()}}
 (out/f'smoke_{args.phase}.json').write_text(json.dumps(report,indent=2));print('RESIDUAL_SMOKE_OK',report,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True
 env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
