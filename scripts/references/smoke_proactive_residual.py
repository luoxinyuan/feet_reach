"""Exercise frozen CEER, nominal-target isolation, and both residual PPO phases."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys, argparse, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--phase',default='train');p.add_argument('--checkpoint');args=p.parse_args()
import torch, hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=['task=G1/G1_proactive_residual','algo=proactive_residual',f'algo.phase={args.phase}','task.num_envs=8','algo.ppo_epochs=1','algo.num_minibatches=1','algo.train_every=8','vecnorm=train','wandb.mode=disabled',f'task.action.low_policy.checkpoint_path={ROOT}/.cache/ceer/checkpoint_final.pt'])
if args.checkpoint: cfg.checkpoint_path=args.checkpoint
if args.phase=='adapt': cfg.vecnorm='eval'
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
 assert set(funcs)=={'left_ee_tracking','contact_jump_cost'},set(funcs)
 nominal=cmd.target_ee_pos_b().clone();reference=act._get_ee_command_reference().clone();before=cmd.left_ee_tracking().clone()
 decoded=act._decode_ee_command(torch.ones(8,3,device=cmd.device))
 torch.testing.assert_close(decoded[:,3:],reference[:,3:])
 torch.testing.assert_close(decoded[:,:3]-reference[:,:3],torch.full_like(nominal,.15*float(torch.tanh(torch.tensor(1.)))))
 cmd.set_root_and_wrist_6d_command(decoded)
 torch.testing.assert_close(cmd.target_ee_pos_b(),nominal);torch.testing.assert_close(cmd.left_ee_tracking(),before)
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
 out=ROOT/'artifacts/proactive_residual';out.mkdir(exist_ok=True,parents=True)
 torch.save({'policy':policy.state_dict(),'vecnorm':vecnorm.state_dict(),'cfg':cfg},out/f'smoke_{args.phase}.pt')
 report={'phase':args.phase,'frozen_ceer_unchanged':True,'metrics':{k:float(v) for k,v in info.items()}}
 (out/f'smoke_{args.phase}.json').write_text(json.dumps(report,indent=2));print('RESIDUAL_SMOKE_OK',report,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True
 env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
