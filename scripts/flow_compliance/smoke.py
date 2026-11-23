"""Real-physics audit of GT reference increments and frozen CEER residual PPO."""
import os,sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type,ExplorationType
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['random','optimized','observable3','observable5'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--updates',type=int,default=2);p.add_argument('--task',default=None,help='Optional compatible task config for data-integration audits');a=p.parse_args()
rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 task=a.task or ('G1/G1_supermimic_surface_'+a.mode)
 cfg=hydra.compose(config_name='train',overrides=[f'task={task}','algo=supermimic_flow_latent','algo.flow_warmup=0',f'task.num_envs={a.num_envs}','wandb.mode=disabled'])
cfg.seed+=rank*10000;OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device=f'cuda:{rank}',distributed=world>1,kit_args='--/plugins/carb.tasking.plugin/threadCount=8').app
try:
 if world>1:torch.distributed.init_process_group('nccl')
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.learning.hierarchical.dense_force_ppo import plain
 env,agent,norm,_=make_env_policy(cfg);env.train();td=env.reset();cmd=env.command_manager
 frozen={k:v.clone() for k,v in env.action_manager.low_policy.state_dict().items()};actor0={k:v.clone() for k,v in plain(agent.actor).state_dict().items()};flow0={k:v.clone() for k,v in plain(agent.flow).state_dict().items()}
 assert plain(agent.flow).horizon==1 and plain(agent.flow).velocity[-1].out_features==32
 rollout=agent.get_rollout_policy('train');public=agent.get_rollout_policy('deploy')
 assert set(public.in_keys)=={'policy','force_gt_'},public.in_keys
 assert td['force_gt_'].shape[-1]==cmd.token_dim*len(cmd.history_offsets)
 assert 'estimator_history_' not in td and not hasattr(agent,'estimator')
 assert cmd.force_apply_world
 audit={'steps':0,'max_force':0.}
 def physics_audit(substep):
  from active_adaptation.utils.math import quat_apply
  offsets=quat_apply(cmd.asset.data.body_quat_w[cmd.rows[:,None],cmd.contact_body],cmd.local_points)
  positions=cmd.asset.data.body_pos_w[cmd.rows[:,None],cmd.contact_body]+offsets
  root=cmd.asset.data.root_pos_w[:,None]
  force=cmd.force_apply_buffer;torque=cmd.torque_apply_buffer
  torch.testing.assert_close(force.sum(1),cmd.force.sum(1),atol=2e-5,rtol=1e-4)
  lhs=(torch.cross(cmd.asset.data.body_pos_w-root,force,dim=-1)+torque).sum(1)
  rhs=torch.cross(positions-root,cmd.force,dim=-1).sum(1)
  torch.testing.assert_close(lhs,rhs,atol=2e-3,rtol=1e-4)
  audit['steps']+=1;audit['max_force']=max(audit['max_force'],cmd.force.norm(dim=-1).max().item())
 env.base_env._pre_step_callbacks.append(physics_audit)
 def verify_prompt(td):
  assert torch.isfinite(td['force_gt_']).all()
  assert cmd.force.norm(dim=-1).sum(-1).max()<=60.0001
  assert cmd.force.norm(dim=-1).max()<=30.0001
  assert not cmd.force[~cmd.active].any()
  assert ((cmd.active.sum(-1)>=(0 if a.mode.startswith('observable') else 1))&(cmd.active.sum(-1)<=cmd.max_contacts)).all()
  assert cmd.contact_bary.min()>=-1e-6
  torch.testing.assert_close(cmd.contact_bary.sum(-1),torch.ones(cmd.num_envs,cmd.max_contacts,device=cmd.device),atol=1e-5,rtol=1e-5)
  torch.testing.assert_close(cmd.local_points,(cmd.contact_triangle*cmd.contact_bary[...,None]).sum(-2),atol=1e-6,rtol=1e-6)
 # Verify low-level runtime cannot access motion labels, even indirectly.
 if a.mode.startswith('observable'):
  manager=env.action_manager;ids=torch.arange(cmd.num_envs,device=cmd.device)
  class ForbiddenMotion:
   def __getattr__(self,name):raise AssertionError('Low-level accessed motion: '+name)
  zero=td.clone();zero['action']=torch.zeros(cmd.num_envs,manager.action_dim,device=cmd.device)
  manager.reset(ids);manager(zero,0);low=manager.low_action.clone();latent=manager.last_latent.clone()
  dataset,motion=cmd.dataset,cmd._motion
  manager.reset(ids);cmd.dataset=ForbiddenMotion();cmd._motion=ForbiddenMotion()
  try:
   manager(zero,0);torch.testing.assert_close(low,manager.low_action,atol=0,rtol=0);torch.testing.assert_close(latent,manager.last_latent,atol=0,rtol=0)
  finally:cmd.dataset,cmd._motion=dataset,motion
  td=env.reset()
 verify_prompt(td)
 assert td['force_gt_'].count_nonzero()==0, 'Unapplied planned contacts leaked to actor at reset'
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'];other=td.clone()
  for key in ['joint_target','priv','priv_critic']:
   if key in other:other[key].fill_(12345.)
  torch.testing.assert_close(x,public(other)['action'],atol=0,rtol=0)
  # Direct increment changes the actor distribution.
  other=td.select(*public.in_keys).clone();other['force_gt_'].add_(.2)
  assert not torch.equal(x,public(other)['action'])
  funcs=env.observation_funcs['policy'].funcs;before={k:f().clone() for k,f in funcs.items()}
  motion=cmd._motion;saved={k:getattr(motion,k).clone() for k in ['joint_pos','joint_vel','body_pos_w','root_pos_w','root_quat_w']}
  for k in saved:getattr(motion,k).fill_(4321.)
  for k,f in funcs.items():torch.testing.assert_close(before[k],f(),atol=0,rtol=0)
  for k,v in saved.items():getattr(motion,k).copy_(v)
 reports=[];start=time.monotonic()
 for update in range(a.updates):
  tic=time.monotonic();samples=[]
  with torch.no_grad(),set_exploration_type(ExplorationType.RANDOM):
   for _ in range(32):
    rollout(td);sample,td=env.step_and_maybe_reset(td);verify_prompt(td)
    agent.critic(sample);agent.critic(sample['next']);sample['next','state_value'][:]=torch.where(sample['next','done'],sample['state_value'],sample['next','state_value'])
    sample['next']=sample['next'].exclude(*rollout.in_keys)
    private=[k for k in sample.keys(True,True) if isinstance(k,str) and k.startswith('_')]
    samples.append(sample.exclude(*private).clone())
  info=agent.train_op(torch.stack(samples,1),norm);assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values())
  info['seconds']=time.monotonic()-tic;reports.append(info);print('UPDATE',rank,update,info['seconds'],flush=True)
 assert any(not torch.equal(v,actor0[k]) for k,v in plain(agent.actor).state_dict().items())
 for k,v in env.action_manager.low_policy.state_dict().items():assert torch.equal(v,frozen[k]),k
 assert all(not p.requires_grad and p.grad is None for p in env.action_manager.low_policy.parameters())
 # Permuting physical contact slots must preserve the complete observation.
 saved_history=cmd.contact_history.clone();saved_fresh=cmd.fresh.clone();saved_tick=cmd.history_tick
 attrs=['contact_body','contact_triangle','contact_bary','local_points','force','active']
 saved={k:getattr(cmd,k).clone() for k in attrs}
 cmd.contact_history.zero_();cmd.fresh.fill_(True);cmd.history_tick=-1
 token1=cmd.surface_contacts().clone()
 for k in attrs:getattr(cmd,k).copy_(saved[k][:,torch.arange(cmd.max_contacts-1,-1,-1,device=cmd.device)])
 cmd.contact_history.zero_();cmd.fresh.fill_(True);cmd.history_tick=-1
 token2=cmd.surface_contacts().clone();torch.testing.assert_close(token1,token2,atol=0,rtol=0)
 for k in attrs:getattr(cmd,k).copy_(saved[k])
 cmd.contact_history.copy_(saved_history);cmd.fresh.copy_(saved_fresh);cmd.history_tick=saved_tick
 assert any(not torch.equal(v,flow0[k]) for k,v in plain(agent.flow).state_dict().items())
 assert audit['steps']>0 and audit['max_force']>0
 if world>1:
  for module in [agent.actor,agent.critic,agent.flow]:
   for value in module.parameters():
    target=value.detach().clone();torch.distributed.broadcast(target,0);torch.testing.assert_close(value,target,atol=0,rtol=0)
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'].clone();agent.load_state_dict(agent.state_dict());y=public(td.select(*public.in_keys).clone())['action'];torch.testing.assert_close(x,y,atol=0,rtol=0)
 # Exercise the actual flow->latent->frozen decoder->physics deployment path.
 with torch.no_grad(),set_exploration_type(ExplorationType.RANDOM):
  for _ in range(32):
   public(td);assert torch.isfinite(td['action']).all()
   _,td=env.step_and_maybe_reset(td)
   assert torch.isfinite(env.command_manager.asset.data.joint_pos).all()
 result=dict(rank=rank,world_size=world,num_envs=a.num_envs,actor_keys=list(public.in_keys),contact_token_dim=cmd.token_dim*len(cmd.history_offsets),force_history=True,physical_force=True,physics_audit=audit,lowlevel_uses_gt_reference=not hasattr(env.action_manager,'goal_interface'),mode=a.mode,target_isolation=True,mesh_contacts=True,permutation_invariant=True,unapplied_contacts_hidden=True,actor_updated=True,flow_updated=True,flow_physics_steps=32,flow_horizon=plain(agent.flow).horizon,flow_output_dim=plain(agent.flow).velocity[-1].out_features,low_level_frozen=True,ddp_synchronized=world>1,checkpoint_roundtrip=True,max_cuda_allocated_GB=torch.cuda.max_memory_allocated()/1e9,wall_seconds=time.monotonic()-start,updates=reports)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2));print('FLOW_SMOKE_PASS',result,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
 if world>1:torch.distributed.destroy_process_group()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
