"""Matched zero-force ablation: learned/zero residual x measured/standing root interface."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'));os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import torch,numpy as np
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
ckpt=ROOT/'outputs/ceer-segmented-jacobian-residual/20261027_173438_ee-root3/wandb/run-20261027_173540-a88dyzbq/files/checkpoint_final.pt'
s=torch.load(ckpt,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(s['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
cfg.checkpoint_path=str(ckpt);cfg.vecnorm='eval';cfg.task.num_envs=128;cfg.seed=42;cfg.wandb.mode='disabled';cfg.eval_render=False;cfg.task.command.guidance_enabled=False;cfg.task.reward={};cfg.task.termination={};cfg.task.max_episode_length=10000;cfg.task.observe_on_reset=True
app=AppLauncher(headless=True,device='cuda:0',kit_args='--/plugins/carb.tasking.plugin/threadCount=16').app
try:
 from scripts.utils.helpers import make_env_policy
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval();cmd=env.command_manager;r=cmd.asset;m=env.action_manager;dev=env.device
 group=torch.arange(128,device=dev)//32
 original=cmd.reset
 def reset(ids):
  original(ids);root=r.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids];root[:,3:7]=root.new_tensor([1,0,0,0]);root[:,7:]=0
  r.write_root_state_to_sim(root,env_ids=ids);r.write_joint_state_to_sim(r.data.default_joint_pos[ids],torch.zeros_like(r.data.default_joint_vel[ids]),env_ids=ids);cmd.joint_pos_boot_protect[ids]=r.data.default_joint_pos[ids]
  for name in ['force','desired','point_force_buffer','point_desired','force_apply_buffer','torque_apply_buffer']:getattr(cmd,name)[ids]=0
  torch.manual_seed(4242)
 callbacks=env.base_env._reset_callbacks;callbacks[:]=[reset if x==original else x for x in callbacks]
 cmd.before_update=lambda:None;cmd.update=lambda:None
 def no_force(substep):
  for name in ['force','desired','point_force_buffer','point_desired','force_apply_buffer','torque_apply_buffer']:getattr(cmd,name).zero_()
  cmd.position_apply_buffer.copy_(r.data.body_pos_w);r.has_external_wrench=False
 callbacks=env.base_env._pre_step_callbacks;callbacks[:]=[no_force if x==cmd.step else x for x in callbacks];assert no_force in callbacks
 low=m.low_inputs
 def inputs(history):
  obs,q=low(history);obs[group>=2,1]=r.data.default_root_state[0,2];obs[group>=2,2:4]=0;return obs,q
 m.low_inputs=inputs
 td=env.reset();env.base_env.episode_length_buf.zero_()
 # Match actuator reset parameters across all four groups.
 lam=m.low_action_manager
 for name,value in vars(lam).items():
  if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==128:
   for j in range(1,4):value[j*32:(j+1)*32].copy_(value[:32])
 start=r.data.root_pos_w.clone();q0=m.reference.q.clone();first=torch.full((128,),float('nan'),device=dev);traces=[];policy=agent.get_rollout_policy('deploy')
 with torch.inference_mode(),set_exploration_type(ExplorationType.MODE):
  for i in range(1500):
   policy(td);raw=td['action'].clone();td['action'][(group%2)==1]=0
   td=env.step(td)['next'];assert not td['done'].any()
   height=r.data.root_pos_w[:,2]-env.scene.env_origins[:,2];tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1));fall=(height<.45)|(tilt>np.pi/3);first[fall&torch.isnan(first)]=(i+1)*.02
   if i%50==0 or i==1499:
    traces.append(dict(t=(i+1)*.02,root=(r.data.root_pos_w-start).cpu().tolist(),height=height.cpu().tolist(),qref_mae=(m.reference.q-q0).abs().mean(-1).cpu().tolist(),ref_speed=m.reference.velocity.abs().mean(-1).cpu().tolist(),raw_abs=raw.abs().mean(-1).cpu().tolist()))
 result={'checkpoint':str(ckpt),'seconds':30,'repeats':32,'groups':{}}
 for j,name in enumerate(['learned_measured_root','zero_residual_measured_root','learned_standing_root','zero_residual_standing_root']):
  sl=slice(j*32,(j+1)*32);result['groups'][name]=dict(survival=float(torch.isnan(first[sl]).float().mean()),first_fall_s=[None if np.isnan(x) else x for x in first[sl].cpu().tolist()],final_root_xy_m=float((r.data.root_pos_w-start)[sl,:2].norm(dim=-1).mean()),final_qref_mae_rad=float((m.reference.q-q0)[sl].abs().mean()),first_second_qref_mae_rad=float(np.array(traces[1]['qref_mae'])[sl].mean()),final_height=float(height[sl].mean()))
 out=ROOT/'artifacts/ee_root3_standing_diagnosis_20261028';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2));(out/'traces.json').write_text(json.dumps(traces));print('DIAGNOSIS_DONE',json.dumps(result),flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
