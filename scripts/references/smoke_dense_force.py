"""Real simulator checks for the three dense-force residual phases, including DDP."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type,ExplorationType
p=argparse.ArgumentParser();p.add_argument('--task',default='G1/G1_ceer_dense_link_compliance');p.add_argument('--output',type=Path,default=ROOT/'artifacts/ceer_dense_link/smoke');p.add_argument('--phase',choices=['teacher','distill','finetune'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--num-envs',type=int,default=16);a=p.parse_args()
rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=[f'task={a.task}','algo=ceer_dense_force',f'algo.phase={a.phase}',f'task.num_envs={a.num_envs}','algo.ppo_epochs=2','algo.num_minibatches=2','wandb.mode=disabled',f'vecnorm={"train" if a.phase=="teacher" else "eval"}'])
if a.checkpoint:cfg.checkpoint_path=str(a.checkpoint.resolve())
cfg.seed+=rank*10000;OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device=f'cuda:{rank}',distributed=world>1,kit_args='--/plugins/carb.tasking.plugin/threadCount=16').app
try:
 if world>1:torch.distributed.init_process_group('nccl')
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.learning.hierarchical.dense_force_ppo import plain
 env,agent,norm,_=make_env_policy(cfg);env.train();td=env.reset();cmd=env.command_manager
 low=env.action_manager.low_policy;frozen={k:v.clone() for k,v in low.state_dict().items()};actor0={k:v.clone() for k,v in plain(agent.actor).state_dict().items()};est0={k:v.clone() for k,v in plain(agent.estimator).state_dict().items()};teacher0={k:v.clone() for k,v in agent.teacher.state_dict().items()}
 rollout=agent.get_rollout_policy('train');public=agent.get_rollout_policy('deploy')
 expected={'policy','force_gt_'} if a.phase=='teacher' else {'policy','estimator_history_'}
 assert set(public.in_keys)==expected,public.in_keys
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=td.select(*public.in_keys).clone();y=td.clone()
  for key in ('joint_target','priv','priv_critic'):
   if key in y:y[key].fill_(12345.)
  if a.phase!='teacher':y['force_gt_'].fill_(999.)
  public(x);public(y);torch.testing.assert_close(x['action'],y['action'],atol=0,rtol=0)
  # Public observation functions cannot read the reference motion.
  functions=env.observation_funcs['policy'].funcs;before={k:f().clone() for k,f in functions.items()}
  motion=cmd._motion;saved={k:getattr(motion,k).clone() for k in ['joint_pos','joint_vel','body_pos_w','root_pos_w','root_quat_w']}
  for k in saved:getattr(motion,k).fill_(4321.)
  for k,f in functions.items():torch.testing.assert_close(before[k],f(),atol=0,rtol=0)
  for k,v in saved.items():getattr(motion,k).copy_(v)
 reports=[]
 for update in range(2):
  samples=[]
  with torch.no_grad(),set_exploration_type(ExplorationType.RANDOM):
   for _ in range(32):
    rollout(td);sample,td=env.step_and_maybe_reset(td)
    assert torch.isfinite(cmd.point_force_buffer).all()
    torch.testing.assert_close(cmd.force_apply_buffer[:,cmd.point_ids],cmd.point_force_buffer)
    assert cmd.point_force_buffer.norm(dim=-1).max()<=30.00001
    agent.critic(sample);agent.critic(sample['next']);sample['next','state_value'][:]=torch.where(sample['next','done'],sample['state_value'],sample['next','state_value'])
    sample['next']=sample['next'].exclude(*rollout.in_keys)
    private=[k for k in sample.keys(True,True) if isinstance(k,str) and k.startswith('_')]
    samples.append(sample.exclude(*private).clone())
  info=agent.train_op(torch.stack(samples,1),norm)
  assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values());reports.append(info)
 assert any(not torch.equal(v,actor0[k]) for k,v in plain(agent.actor).state_dict().items())
 for k,v in low.state_dict().items():assert torch.equal(v,frozen[k]),k
 assert all(not p.requires_grad and p.grad is None for p in low.parameters())
 for k,v in agent.teacher.state_dict().items():assert torch.equal(v,teacher0[k]),k
 if a.phase!='teacher':assert any(not torch.equal(v,est0[k]) for k,v in plain(agent.estimator).state_dict().items())
 if world>1:
  for module in [agent.actor,agent.estimator]:
   for value in module.parameters():
    target=value.detach().clone();torch.distributed.broadcast(target,0);torch.testing.assert_close(value,target,atol=0,rtol=0)
 # Restore checkpoint and verify exact inference / normalizer preservation.
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  x=public(td.select(*public.in_keys).clone())['action'].clone()
  agent.load_state_dict(agent.state_dict())
  y=public(td.select(*public.in_keys).clone())['action'];torch.testing.assert_close(x,y,atol=0,rtol=0)
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 result=dict(phase=a.phase,rank=rank,world_size=world,public_keys=list(public.in_keys),force_dim=td['force_gt_'].shape[-1],history_dim=td['estimator_history_'].shape[-1],updates=reports,low_level_frozen=True,teacher_frozen=True,target_isolation=True,checkpoint_roundtrip=True,actor_updated=True,estimator_updated=a.phase!='teacher',ddp_synchronized=world>1)
 (out/f'{a.phase}_{world}gpu_rank{rank}.json').write_text(json.dumps(result,indent=2))
 if rank==0:torch.save(dict(policy=agent.state_dict(),vecnorm=norm.state_dict(),cfg=cfg),out/f'{a.phase}.pt')
 print('DENSE_SMOKE_PASS',json.dumps(result),flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
 if world>1:torch.distributed.destroy_process_group()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
