"""Matched trajectory diagnostic: GT prompt, zero prompt, oracle-reference control."""
import os,sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
from tensordict import TensorDict
from torchrl.envs.transforms import VecNorm
from torchrl.envs.utils import set_exploration_type,ExplorationType
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--clip',type=int,default=22);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'videos').mkdir(exist_ok=True)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
cfg.checkpoint_path=str(a.checkpoint.resolve());cfg.vecnorm='eval';cfg.task.num_envs=3;cfg.wandb.mode='disabled';cfg.eval_render=True;cfg.headless=True;cfg.seed=20261106;cfg.task.viewer.resolution=[960,540]
cfg.task.command.dataset.resample_on_reset=False;cfg.task.command.dataset.fix_ds=0;cfg.task.command.dataset.fix_motion_id=a.clip
# Restore exactly the checkpoint's normalization statistics (no adaptation in eval).
def load_norm(self,saved):
 s=TensorDict(saved).flatten_keys(self.SEP);now=self._td.flatten_keys(self.SEP)
 for key in now.keys():
  assert key in s and now[key].shape==s[key].shape,(key,now[key].shape)
  now[key].copy_(s[key])
 self._td=now.unflatten_keys(self.SEP)
VecNorm.set_extra_state=load_norm
saved_norm=state['vecnorm'];del state
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=True,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/omni.tbb.globalcontrol/maxThreadCount=8').app
try:
 from scripts.utils.helpers import make_env_policy
 from scripts.compliance_eval.visual import Video
 from active_adaptation.utils.math import quat_apply_inverse
 torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
 env,agent,norm,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval();cmd=env.command_manager;robot=cmd.asset;manager=env.action_manager
 expected=TensorDict(saved_norm['_extra_state']).flatten_keys(norm.SEP);actual=norm._td.flatten_keys(norm.SEP)
 for k in ('policy_sum','policy_ssq','policy_count'):torch.testing.assert_close(expected[k].to(actual[k]),actual[k],atol=0,rtol=0)
 cmd.zero_init_prob=1.
 td=env.reset();env.base_env.episode_length_buf.zero_();assert (cmd._motion.motion_id[:,0]==a.clip).all()
 # Same initial robot, delay and actuation filters in all three diagnostic arms.
 low=manager.low_action_manager
 for attr in ['alpha','delay','boot_delay']:
  if hasattr(low,attr):getattr(low,attr)[:]=getattr(low,attr)[0].clone()
 if hasattr(low,'alpha_jit_scale'):low.alpha_jit_scale=None
 policy=agent.get_rollout_policy('deploy');assert set(policy.in_keys)=={'policy','force_gt_'}
 from pxr import UsdGeom
 import omni.usd
 stage=omni.usd.get_context().get_stage()
 for i in [1,2]:UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{i}')).MakeInvisible()
 renderer=Video(env,cmd,a.output/'videos'/f'clip{a.clip:05d}_teacher_prompt.mp4',True,prompt_only=True)
 jointids=manager.joint_ids;refids=cmd.reference_target_ids;hands=[robot.body_names.index(s+'_hand_mimic') for s in ['left','right']];mhands=[cmd.dataset.body_names.index(s+'_hand_mimic') for s in ['left','right']]
 record={};steps=int(cmd.lengths[0])-1;initial=robot.data.joint_pos[:,jointids].clone();fall=torch.full((3,),-1,device=env.device)
 def save(k,x):record.setdefault(k,[]).append(x.detach().cpu().numpy())
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  for step in range(steps+1):
   q=robot.data.joint_pos[:,jointids];target=cmd._motion.joint_pos[:,0,refids]
   actual_h=quat_apply_inverse(robot.data.root_quat_w[:,None].expand(-1,2,-1),robot.data.body_pos_w[:,hands]-robot.data.root_pos_w[:,None])
   target_h=quat_apply_inverse(cmd._motion.root_quat_w[:,0,None].expand(-1,2,-1),cmd._motion.body_pos_w[:,0,mhands]-cmd._motion.root_pos_w[:,0,None])
   save('q',q);save('target_q',target);save('reference_q',manager.reference.q);save('action',manager.applied_action);save('hands_b',actual_h);save('target_hands_b',target_h);save('prompt',cmd.point_force_buffer);save('force_obs',td['force_gt_']);save('physical_force',cmd.force_apply_buffer);save('motion_step',cmd._motion.step[:,0]);save('low_action',manager.low_action)
   fell=(robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]<.3)|(robot.data.projected_gravity_b[:,2]>-np.cos(np.deg2rad(75)))
   fall=torch.where((fall<0)&fell,torch.full_like(fall,step),fall)
   if step%2==0:renderer.frame(step*.02,f'clip {a.clip} | trained residual',bool(fall[0]>=0))
   if step==steps:break
   # Check input alignment on every frame before consuming the prompt.
   expected_force=quat_apply_inverse(robot.data.root_quat_w[:,None].expand(-1,37,-1),cmd.point_force_buffer).flatten(1)/30.
   torch.testing.assert_close(td['force_gt_'],expected_force,atol=1e-7,rtol=1e-6)
   td['force_gt_'][1]=0
   counter=td.select(*policy.in_keys).clone();counter['force_gt_'][0]=0;policy(counter)
   policy(td)
   save('matched_force_action_difference',(td['action'][0].tanh()-counter['action'][0].tanh()))
   save('actor_force_input',td['force_gt_'])
   nxt=cmd.dataset.get_slice(None,torch.minimum(cmd.t+1,cmd.lengths-1),1).joint_pos[:,0,refids]
   ratio=((nxt[2]-manager.reference.q[2])/(env.step_dt*manager.reference.speed)).clamp(-.999,.999)
   td['action'][2]=torch.atanh(ratio)
   td=env.step(td)['next']
   assert cmd.force_apply_buffer.count_nonzero()==0
 renderer.close();arrays={k:np.stack(v) for k,v in record.items()};np.savez_compressed(a.output/'traces.npz',**arrays)
 results=[]
 for i,label in enumerate(['trained_prompt','zero_prompt','oracle_reference']):
  q=arrays['q'][:,i];ref=arrays['reference_q'][:,i];target=arrays['target_q'][:,i];h=arrays['hands_b'][:,i];th=arrays['target_hands_b'][:,i]
  results.append(dict(case=label,joint_mae_rad=float(abs(q-target).mean()),reference_target_mae_rad=float(abs(ref-target).mean()),lowlevel_reference_mae_rad=float(abs(q-ref).mean()),mean_joint_excursion_rad=float(np.ptp(q,axis=0).mean()),target_mean_joint_excursion_rad=float(np.ptp(target,axis=0).mean()),hand_bbox_diagonal_m=np.linalg.norm(np.ptp(h,axis=0),axis=-1).tolist(),target_hand_bbox_diagonal_m=np.linalg.norm(np.ptp(th,axis=0),axis=-1).tolist(),ee_relative_error_m=np.linalg.norm(h-th,axis=-1).mean(0).tolist(),action_abs_mean=float(abs(arrays['action'][1:,i]).mean()),first_fall_step=int(fall[i])))
 result=dict(checkpoint=str(a.checkpoint.resolve()),clip=a.clip,frames=steps+1,duration_s=steps*.02,actor_inputs=list(policy.in_keys),normalizer_verified=True,force_history=False,physical_force_max_N=float(np.linalg.norm(arrays['physical_force'],axis=-1).max()),prompt_mean_total_N=float(np.linalg.norm(arrays['prompt'][:,0],axis=-1).sum(-1).mean()),joint_names=manager.joint_names,cases=results)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('EVAL_RESULT',json.dumps(result),flush=True)
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:sys.stdout.flush();sys.stderr.flush();os._exit(0)
