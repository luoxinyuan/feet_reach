"""Finite end-to-end check for each PPO phase, geometry and contact feedback."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--phase',default='train');p.add_argument('--checkpoint');args=p.parse_args()
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=['task=G1/G1_proactive_ee',f'+exp={args.phase}','task.num_envs=8','algo.symmetry_augmentation=false','algo.geometry_points=128','algo.in_keys=[policy,priv,joint_target,priv_critic,geometry_]','algo.ppo_epochs=1','algo.num_minibatches=1','algo.train_every=8','wandb.mode=disabled'])
if args.checkpoint:cfg.checkpoint_path=args.checkpoint
OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device='cuda:0').app
try:
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.utils.math import quat_apply_inverse
 from active_adaptation.learning.modules.pointnet import GeometryPointNet
 env,policy,vecnorm,_=make_env_policy(cfg);env.train();carry=env.reset();cmd=env.command_manager
 assert carry['geometry_'].shape==(8,512)
 raw=cmd.geometry_cloud();torch.testing.assert_close(carry['geometry_'],raw)
 expected=quat_apply_inverse(cmd.asset.data.root_quat_w,cmd.target_ee_world()-cmd.asset.data.root_pos_w)
 torch.testing.assert_close(cmd.target_ee_pos_b(),expected)
 error=(cmd.asset.data.body_pos_w[:,cmd.hand_asset]-cmd.target_ee_world()).norm(dim=-1).max().item()
 assert error<.002,error
 xyz=torch.tensor([.4,.2,.1],device=cmd.device);cmd.set_target_ee_pos_b(xyz)
 torch.testing.assert_close(cmd.target_ee_pos_b(),xyz.expand(8,-1),atol=1e-6,rtol=1e-5);cmd.set_target_ee_pos_b(None)
 # A reset must randomize physical wall and geometry together.
 cmd.no_wall_prob=1.;carry=env.reset();assert not carry['geometry_'].any()
 assert ((cmd.wall.data.root_pos_w[:,0]-env.scene.env_origins[:,0])>2.).all()
 cmd.no_wall_prob=0.;carry=env.reset();assert carry['geometry_'].reshape(8,128,4)[:,:,3].any()
 torch.testing.assert_close(cmd.wall.data.root_pos_w[:,0]-env.scene.env_origins[:,0]-.025,cmd.wall_x,atol=1e-6,rtol=1e-5)
 # Point-order invariance, finite empty-cloud output and gradient flow.
 net=GeometryPointNet(128).to(cmd.device);cloud=carry['geometry_'].clone().requires_grad_();z=net(cloud)
 perm=torch.randperm(128,device=cmd.device)
 torch.testing.assert_close(z,net(cloud.reshape(8,128,4)[:,perm].flatten(1)))
 z.square().mean().backward();assert cloud.grad[:,:].abs().sum()>0
 assert torch.isfinite(net(torch.zeros_like(cloud))).all()
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
 if args.phase in ('train','finetune'):
  actor=policy.actor_teacher if args.phase=='train' else policy.actor_student
  grads=[p.grad.abs().sum().item() for m in actor.modules() if isinstance(m,GeometryPointNet) for p in m.parameters() if p.grad is not None]
  assert sum(grads)>0, 'Actor PointNet has no PPO gradient'
 if args.phase!='train':
  public=carry.select('policy','geometry_','is_init',strict=False).clone();rollout(public);assert public['action'].shape==(8,29)
 # Check actual collision force by holding root/legs and advancing the nominal
 # reference with the existing PD controller. This is NOT a trained evaluation.
 carry=env.reset();peak=0.;jump=0.
 with torch.no_grad():
  for i in range(180):
   manager=env.action_manager
   target=cmd._motion.joint_pos[:,0,cmd.joint_idx_motion]
   carry['action']=(target-manager.default_joint_pos[:,manager.joint_ids])/manager.action_scaling
   td,carry=env.step_and_maybe_reset(carry)
   peak=max(peak,cmd.peak_force.max().item());jump=max(jump,cmd.peak_rate.max().item())
 assert peak>1.,('No physical hand contact measured',peak)
 assert jump>0
 output=ROOT/'artifacts/ee_reach';output.mkdir(parents=True,exist_ok=True)
 report=dict(phase=args.phase,initial_error_m=error,force_peak_N=peak,force_rate_peak_N_s=jump,metrics={k:float(v) for k,v in info.items()})
 (output/f'smoke_{args.phase}.json').write_text(json.dumps(report,indent=2))
 torch.save({'policy':policy.state_dict(),'vecnorm':vecnorm.state_dict(),'cfg':cfg},output/f'smoke_{args.phase}.pt')
 print('PROACTIVE_EE_SMOKE_OK',report,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True
 env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
