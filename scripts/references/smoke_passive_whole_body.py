import os
os.environ['ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE']='1'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,hydra
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
with hydra.initialize_config_dir(config_dir=str(ROOT/'cfg'),version_base=None):
 cfg=hydra.compose(config_name='train',overrides=['task=G1/G1_passive_whole_body','algo=passive_whole_body','task.num_envs=8','algo.ppo_epochs=1','algo.num_minibatches=1','algo.train_every=16','wandb.mode=disabled'])
OmegaConf.resolve(cfg)
app=AppLauncher(headless=True,device='cuda:0').app
try:
 from scripts.utils.helpers import make_env_policy
 env,agent,vecnorm,_=make_env_policy(cfg);env.train();td=env.reset();cmd=env.command_manager
 assert cmd.dataset.counts==[7692]
 assert len(cmd.joint_idx_asset)==29
 with torch.no_grad():
  teacher=agent.get_rollout_policy('train');steps=[]
  for i in range(16):
   td=teacher(td);sample,td=env.step_and_maybe_reset(td)
   assert torch.isfinite(cmd.force).all()
   assert cmd.force.norm(dim=-1).max()<30.001
   assert cmd.force.norm(dim=-1).sum(-1).max()<60.001
   assert (cmd.force_apply_buffer.norm(dim=-1)>1e-5).sum(-1).max()<=3
   assert cmd.torque_apply_buffer.count_nonzero()==0
   agent.critic(sample);agent.critic(sample['next'])
   sample['next','state_value'][:]=torch.where(sample['next','done'],sample['state_value'],sample['next','state_value'])
   sample['next']=sample['next'].exclude(*teacher.in_keys)
   private=[k for k in sample.keys(True,True) if isinstance(k,str) and k.startswith('_')]
   steps.append(sample.exclude(*private,'priv_pred','priv_feature').clone())
  nxt=cmd.dataset.get_slice(None,torch.minimum(cmd.t+1,cmd.lengths-1),1)
  torch.testing.assert_close(cmd.anchor,nxt.body_pos_w[:,0,cmd.motion_point_ids]+env.scene.env_origins[:,None])
  # Force a contact replacement; it must fade out before selecting new points.
  cmd.remaining.zero_();cmd.before_update()
  for i in range(60):
   prev=cmd.force_apply_buffer.clone();cmd.step(0)
   assert (cmd.force_apply_buffer-prev).norm(dim=-1).max()<=150*env.physics_dt+.001
   if i%4==0:cmd.before_update()
  agent.cfg.phase='adapt';student=agent.get_rollout_policy('deploy')
  public=td.select('policy').clone();student(public)
  private=td.clone();private['joint_target'].fill_(1000);private['priv'].fill_(-1000);student(private)
  torch.testing.assert_close(public['loc'],private['loc'])
  assert list(student.in_keys)==['policy']
  agent.cfg.phase='train'
 agent.step_schedule(.5,1);info=agent.train_op(torch.stack(steps,1),vecnorm)
 assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values())
 phase_info={}
 agent.load_state_dict(agent.state_dict())
 for phase in ['adapt','finetune']:
  agent.cfg.phase=phase;td=env.reset();rollout=agent.get_rollout_policy('deploy');samples=[]
  assert list(rollout.in_keys)==['policy']
  with torch.no_grad():
   for _ in range(16):
    td=rollout(td);sample,td=env.step_and_maybe_reset(td)
    agent.critic(sample);agent.critic(sample['next'])
    sample['next','state_value'][:]=torch.where(sample['next','done'],sample['state_value'],sample['next','state_value'])
    sample['next']=sample['next'].exclude(*rollout.in_keys)
    private=[k for k in sample.keys(True,True) if isinstance(k,str) and k.startswith('_')]
    samples.append(sample.exclude(*private,'priv_pred','priv_feature').clone())
  result=agent.train_op(torch.stack(samples,1),vecnorm)
  assert all(torch.isfinite(torch.as_tensor(v)).all() for v in result.values())
  phase_info[phase]={k:float(v) for k,v in result.items()}
 agent.cfg.phase='train'
 report={'ppo':{k:float(v) for k,v in info.items()},'max_force_N':cmd.force.norm(dim=-1).max().item(),'max_contacts':int((cmd.force_apply_buffer.norm(dim=-1)>1e-5).sum(-1).max()),'student_target_isolated':True,'slew_verified':True,'teacher':True}
 out=ROOT/'artifacts/passive_whole_body';out.mkdir(exist_ok=True,parents=True);(out/'smoke.json').write_text(json.dumps(report,indent=2));print('SMOKE_PASS',report,flush=True)
 (out/'phase_smoke.json').write_text(json.dumps(phase_info,indent=2))
 (out/'smoke_run/train').mkdir(parents=True,exist_ok=True)
 torch.save({'policy':agent.state_dict(),'vecnorm':vecnorm.state_dict(),'cfg':cfg},out/'smoke_run/train/checkpoint_final.pt')
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();os._exit(1)
else:app.close()
