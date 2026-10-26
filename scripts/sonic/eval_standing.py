"""Trained residual actor: unforced nominal standing, no automatic resets."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
cfg.checkpoint_path=str(args.checkpoint);cfg.vecnorm='eval';cfg.task.num_envs=64;cfg.seed=42
cfg.task.command.guidance_enabled=False;cfg.wandb.mode='disabled';cfg.eval_render=False;cfg.headless=True
cfg.task.command.dataset.mem_paths=['passive_whole_body_full/val'];del state
app=AppLauncher(headless=True,device='cuda:0',kit_args='--/plugins/carb.tasking.plugin/threadCount=16').app
try:
 from scripts.utils.helpers import make_env_policy
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 torch.manual_seed(42)
 env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval()
 cmd=env.command_manager;robot=cmd.asset;manager=env.action_manager
 original=cmd.reset
 def reset_standing(ids):
  original(ids)
  root=robot.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids];root[:,7:]=0
  robot.write_root_state_to_sim(root,env_ids=ids)
  robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],torch.zeros_like(robot.data.default_joint_vel[ids]),env_ids=ids)
  cmd.joint_pos_boot_protect[ids]=robot.data.default_joint_pos[ids]
  cmd.force[ids]=0;cmd.desired[ids]=0;cmd.force_apply_buffer[ids]=0;cmd.torque_apply_buffer[ids]=0
 callbacks=env.base_env._reset_callbacks
 callbacks[:]=[reset_standing if cb==original else cb for cb in callbacks]
 assert reset_standing in callbacks
 policy=agent.get_rollout_policy('deploy');assert list(policy.in_keys)==['policy']
 torch.manual_seed(42);td=env.reset();root0=robot.data.root_pos_w.clone();q0=robot.data.joint_pos.clone()
 alive=torch.ones(64,dtype=torch.bool,device=env.device);duration=torch.zeros(64,device=env.device);maxdrift=torch.zeros_like(duration)
 minheight=10.;maxtilt=0.;speed_sum=0.;q_sum=0.;ref_sum=0.;count=0;traces=[]
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  for step in range(500):
   td=policy(td);td=env.step(td)['next']
   assert cmd.force_apply_buffer.count_nonzero()==0
   height=robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
   tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
   drift=(robot.data.root_pos_w[:,:2]-root0[:,:2]).norm(dim=-1)
   minheight=min(minheight,height[alive].min().item());maxtilt=max(maxtilt,tilt[alive].max().item())
   maxdrift=torch.where(alive,torch.maximum(maxdrift,drift),maxdrift)
   speed_sum+=robot.data.root_lin_vel_w[alive,:2].norm(dim=-1).sum().item()
   q_sum+=(robot.data.joint_pos-q0).abs().mean(-1)[alive].sum().item()
   ref_sum+=(manager.reference.q-q0[:,manager.joint_ids]).abs().mean(-1)[alive].sum().item()
   count+=alive.sum().item();duration+=alive.float()*.02
   alive&=(height>=.45)&(tilt<=torch.pi/3)
   if not alive.any():break
 report=dict(checkpoint=str(args.checkpoint),num_envs=64,horizon_s=10.,physical_survival_fraction=alive.float().mean().item(),
  mean_duration_s=duration.mean().item(),max_root_drift_mean_m=maxdrift.mean().item(),max_root_drift_p95_m=maxdrift.quantile(.95).item(),
  min_root_height_m=minheight,max_tilt_rad=maxtilt,mean_root_speed_m_s=speed_sum/count,mean_joint_departure_rad=q_sum/count,mean_self_reference_departure_rad=ref_sum/count,
  notes='Nominal measured standing reset; deterministic trained student; zero applied force; no automatic reset; AMASS tracking termination ignored; stop counting at physical fall.')
 (args.output/'summary.json').write_text(json.dumps(report,indent=2));print('STANDING_EVAL_COMPLETE',report,flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
