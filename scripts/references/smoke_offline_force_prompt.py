"""Real-physics residual PPO audit: tokens affect actor, never physics directly."""
import os,sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type,ExplorationType
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--updates',type=int,default=2);p.add_argument('--reserve-cache-frames',type=int,default=0);p.add_argument('--physical',action='store_true');a=p.parse_args()
m=json.loads((a.cache/'manifest.json').read_text());rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 task='G1/G1_ceer_offline_force_physical_teacher' if a.physical else 'G1/G1_ceer_offline_force_prompt_teacher'
 cfg=hydra.compose(config_name='train',overrides=[f'task={task}','algo=ceer_force_prompt_teacher',f'task.num_envs={a.num_envs}','wandb.mode=disabled',f'task.command.guidance_cache={a.cache.resolve()}','task.command.dataset.mem_paths='+json.dumps(m['dataset_paths'],separators=(',',':'))])
cfg.seed+=rank*10000;OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device=f'cuda:{rank}',distributed=world>1,kit_args='--/plugins/carb.tasking.plugin/threadCount=8').app
try:
 if world>1:torch.distributed.init_process_group('nccl')
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.learning.hierarchical.dense_force_ppo import plain
 env,agent,norm,_=make_env_policy(cfg);env.train();td=env.reset();cmd=env.command_manager
 reserve=torch.zeros(a.reserve_cache_frames,37,3,device=cmd.device) if a.reserve_cache_frames else None
 frozen={k:v.clone() for k,v in env.action_manager.low_policy.state_dict().items()};actor0={k:v.clone() for k,v in plain(agent.actor).state_dict().items()}
 rollout=agent.get_rollout_policy('train');public=agent.get_rollout_policy('deploy')
 assert set(public.in_keys)=={'policy','force_gt_'},public.in_keys
 assert td['force_gt_'].shape[-1]==111
 assert 'estimator_history_' not in td and not hasattr(agent,'estimator')
 assert cmd.force_apply_world==a.physical
 physics_audit={'steps':0,'max_body_force_N':0.}
 def verify_physics(substep):
  if not a.physical:return
  from active_adaptation.utils.math import quat_apply
  origin=cmd.asset.data.body_pos_w
  points=origin[:,cmd.prompt_body_ids]+quat_apply(cmd.asset.data.body_quat_w[:,cmd.prompt_body_ids],cmd.prompt_offsets[None].expand(cmd.num_envs,-1,-1))
  root=cmd.asset.data.root_pos_w[:,None]
  torch.testing.assert_close(cmd.position_apply_buffer,origin,atol=0,rtol=0)
  torch.testing.assert_close(cmd.force_apply_buffer.sum(1),cmd.point_force_buffer.sum(1),atol=2e-5,rtol=1e-4)
  torque=(torch.cross(origin-root,cmd.force_apply_buffer,dim=-1)+cmd.torque_apply_buffer).sum(1)
  desired=torch.cross(points-root,cmd.point_force_buffer,dim=-1).sum(1)
  torch.testing.assert_close(torque,desired,atol=2e-3,rtol=1e-4)
  physics_audit['steps']+=1
  physics_audit['max_body_force_N']=max(physics_audit['max_body_force_N'],cmd.force_apply_buffer.norm(dim=-1).max().item())
 env.base_env._pre_step_callbacks.append(verify_physics)
 def verify_prompt(td):
  from active_adaptation.utils.math import quat_apply_inverse
  mid=cmd._motion.motion_id[:,0].long();step=cmd._motion.step[:,0].long()
  expected=cmd.cache_forces[cmd.cache_offsets[mid]+step].clone();expected[cmd.t>=cmd.lengths-1]=0
  torch.testing.assert_close(cmd.point_force_buffer,expected,atol=0,rtol=0)
  q=cmd.asset.data.root_quat_w[:,None].expand(-1,37,-1)
  torch.testing.assert_close(td['force_gt_'],quat_apply_inverse(q,expected).flatten(1)/30.,atol=0,rtol=0)
  if not a.physical:assert cmd.force_apply_buffer.count_nonzero()==0 and cmd.torque_apply_buffer.count_nonzero()==0
  assert cmd.point_force_buffer.norm(dim=-1).sum(-1).max()<=60.0001
 verify_prompt(td)
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'];other=td.clone()
  for key in ['joint_target','priv','priv_critic']:
   if key in other:other[key].fill_(12345.)
  torch.testing.assert_close(x,public(other)['action'],atol=0,rtol=0)
  # Force prompt changes the actor distribution; there is no stale force/history.
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
 if world>1:
  for module in [agent.actor,agent.critic]:
   for value in module.parameters():
    target=value.detach().clone();torch.distributed.broadcast(target,0);torch.testing.assert_close(value,target,atol=0,rtol=0)
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'].clone();agent.load_state_dict(agent.state_dict());y=public(td.select(*public.in_keys).clone())['action'];torch.testing.assert_close(x,y,atol=0,rtol=0)
 if a.physical:assert physics_audit['steps']>0 and physics_audit['max_body_force_N']>0
 result=dict(rank=rank,world_size=world,num_envs=a.num_envs,actor_keys=list(public.in_keys),force_dim=111,force_history=False,physical_force=a.physical,physics_audit=physics_audit,target_isolation=True,force_index_alignment=True,actor_updated=True,low_level_frozen=True,ddp_synchronized=world>1,checkpoint_roundtrip=True,max_cuda_allocated_GB=torch.cuda.max_memory_allocated()/1e9,wall_seconds=time.monotonic()-start,updates=reports)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2));print('PROMPT_SMOKE_PASS',result,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
 if world>1:torch.distributed.destroy_process_group()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
