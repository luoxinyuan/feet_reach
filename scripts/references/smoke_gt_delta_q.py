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
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--updates',type=int,default=2);a=p.parse_args()
m=json.loads((a.cache/'manifest.json').read_text());rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 task='G1/G1_ceer_gt_delta_q_teacher'
 cfg=hydra.compose(config_name='train',overrides=[f'task={task}','algo=ceer_gt_delta_q_teacher',f'task.num_envs={a.num_envs}','wandb.mode=disabled',f'task.command.guidance_cache={a.cache.resolve()}','task.command.dataset.mem_paths='+json.dumps(m['dataset_paths'],separators=(',',':'))])
cfg.seed+=rank*10000;OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device=f'cuda:{rank}',distributed=world>1,kit_args='--/plugins/carb.tasking.plugin/threadCount=8').app
try:
 if world>1:torch.distributed.init_process_group('nccl')
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.learning.hierarchical.dense_force_ppo import plain
 env,agent,norm,_=make_env_policy(cfg);env.train();td=env.reset();cmd=env.command_manager
 frozen={k:v.clone() for k,v in env.action_manager.low_policy.state_dict().items()};actor0={k:v.clone() for k,v in plain(agent.actor).state_dict().items()}
 rollout=agent.get_rollout_policy('train');public=agent.get_rollout_policy('deploy')
 assert set(public.in_keys)=={'policy','delta_q_gt_'},public.in_keys
 assert td['delta_q_gt_'].shape[-1]==29
 assert 'estimator_history_' not in td and not hasattr(agent,'estimator')
 assert not cmd.force_apply_world
 def verify_prompt(td):
  current=cmd.dataset.get_slice(None,cmd.t,1).joint_pos[:,0,cmd.reference_target_ids].float()
  nxt=cmd.dataset.get_slice(None,torch.minimum(cmd.t+1,cmd.lengths-1),1).joint_pos[:,0,cmd.reference_target_ids].float()
  expected=nxt-current
  expected[cmd.t>=cmd.lengths-1]=0
  torch.testing.assert_close(cmd.delta_q_buffer,expected,atol=0,rtol=0)
  torch.testing.assert_close(td['delta_q_gt_'],expected/cmd.delta_q_scale,atol=0,rtol=0)
  assert cmd.force_apply_buffer.count_nonzero()==0 and cmd.torque_apply_buffer.count_nonzero()==0
 # Explicit end-of-clip and reset checks, including heterogeneous clip lengths.
 saved_t=cmd.t.clone()
 cmd.t.copy_(cmd.lengths-1)
 cmd._motion=cmd.dataset.get_slice(None,cmd.t,steps=cmd.future_steps)
 cmd._solve()
 assert cmd.delta_q_buffer.count_nonzero()==0
 cmd.t.copy_(saved_t)
 td=env.reset()
 verify_prompt(td)
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'];other=td.clone()
  for key in ['joint_target','priv','priv_critic']:
   if key in other:other[key].fill_(12345.)
  torch.testing.assert_close(x,public(other)['action'],atol=0,rtol=0)
  # Direct increment changes the actor distribution.
  other=td.select(*public.in_keys).clone();other['delta_q_gt_'].add_(.2)
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
 result=dict(rank=rank,world_size=world,num_envs=a.num_envs,actor_keys=list(public.in_keys),delta_q_dim=29,force_history=False,physical_force=False,target_isolation=True,delta_q_index_alignment=True,terminal_delta_zero=True,actor_updated=True,low_level_frozen=True,ddp_synchronized=world>1,checkpoint_roundtrip=True,max_cuda_allocated_GB=torch.cuda.max_memory_allocated()/1e9,wall_seconds=time.monotonic()-start,updates=reports)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2));print('DELTA_Q_SMOKE_PASS',result,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
 if world>1:torch.distributed.destroy_process_group()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
