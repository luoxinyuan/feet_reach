"""Native CEER observations versus residual-wrapper substitutions, no external force."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--clip',type=int,default=22);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(s['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
cfg.checkpoint_path=None;cfg.vecnorm=None;cfg.task.num_envs=11;cfg.task.randomization={};cfg.eval_render=False;cfg.headless=True;cfg.seed=20261106
cfg.task.command.dataset=dict(mem_paths=[str(ROOT/'artifacts/offline_force_baseline/train37_budget60/dataset0')],path_weights=[1.],sample_once=True,resample_on_reset=False,fix_ds=0,fix_motion_id=a.clip,position_z_offset=.03540915,clamp_joint_targets=False,buffer_float_dtype='float16')
for k in cfg.task.command.init_noise:cfg.task.command.init_noise[k]=0.
for group in cfg.task.observation.values():
 for term in group.values():
  if term and 'noise_std' in term:term.noise_std=0.
cfg.task.command.external_force_enabled=False
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=False,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/omni.tbb.globalcontrol/maxThreadCount=8').app
try:
 from scripts.utils.helpers import make_env_policy
 from active_adaptation.utils.ceer_residual import FrozenCEER,TARGET_JOINT_NAMES
 from active_adaptation.envs.mdp.ceer_residual import CEERHistory
 from active_adaptation.utils.sonic import JOINT_NAMES
 from active_adaptation.utils.usd_kinematics import USDKinematics
 from active_adaptation.utils.math import quat_apply_inverse
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
 env,agent,norm,_=make_env_policy(cfg);agent.load_state_dict(s['policy']);agent.eval();env.eval();env.base_env.eval()
 low=FrozenCEER(a.checkpoint,env.device);cmd=env.command_manager;robot=cmd.asset;act=env.action_manager
 assert act.joint_names==JOINT_NAMES
 ids=act.joint_ids;order=[cmd.dataset.joint_names.index(n) for n in JOINT_NAMES];to_target=[JOINT_NAMES.index(n) for n in TARGET_JOINT_NAMES]
 assert cmd.dataset.joint_names==TARGET_JOINT_NAMES
 cmd.zero_init_prob=1.;cmd.set_external_force_enabled(False)
 history=CEERHistory(11,env.device)
 def post(substep):history.physics_q[:,substep%2]=robot.data.joint_pos[:,ids]
 env.base_env._post_step_callbacks.append(post)
 td=env.reset();env.base_env.episode_length_buf.zero_();history.reset(torch.arange(11,device=env.device),robot.data.joint_pos[:,ids]);cmd.set_external_force_enabled(False)
 for name in ['alpha','delay','boot_delay']:
  getattr(act,name)[:]=getattr(act,name)[0].clone()
 act.alpha_jit_scale=None
 fk=USDKinematics(robot.cfg.spawn.usd_path,JOINT_NAMES,env.device)
 policy=agent.get_rollout_policy('eval');rec={};equiv=0.;shadow={};falls=torch.full((11,),-1,device=env.device)
 def save(k,v):rec.setdefault(k,[]).append(v.detach().cpu().numpy())
 labels=['native','native_measured_root5','native_repeat_q','native_fk_wrists','full_wrapper','native_force5','wrapper_force30','wrapper_native_root','native_teacher_truepriv','native_student_truepriv','native_teacher_estpriv']
 nsteps=int(cmd.lengths[0])-1
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  for step in range(nsteps+1):
   q=robot.data.joint_pos[:,ids];target=cmd._motion.joint_pos[:,0,order]
   save('q',q);save('target_q',target);save('root_pos',robot.data.root_pos_w-env.scene.env_origins);save('target_root_pos',cmd._motion.root_pos_w[:,0])
   fell=(robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]<.3)|(robot.data.projected_gravity_b[:,2]>-.2588);falls=torch.where((falls<0)&fell,step,falls)
   if step==nsteps:break
   native=td['policy'].clone();jt=td['joint_target'].reshape(11,5,29).clone()
   same=TensorDict({'policy':(native-low.policy_loc)/low.policy_scale,'joint_target':(jt.flatten(1)-low.joint_loc)/low.joint_scale},[11]);policy(same)
   f=low(native,jt);equiv=max(equiv,(same['action']-f).abs().max().item())
   h=history.push(q)
   measured=torch.cat((robot.data.root_pos_w[:,2:3]-env.scene.env_origins[:,2:3],robot.data.root_lin_vel_b[:,:2],torch.ones(11,1,device=env.device),torch.zeros(11,1,device=env.device),torch.full((11,1),5.,device=env.device)),-1)
   wrist=fk.interface(target)
   wrapper=torch.cat((torch.full((11,1),max(0,1-step/25),device=env.device),measured,wrist,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,h,act.action_buf[:,:3].flatten(1)),-1)
   # Shadow input errors on the SAME states isolate preprocessing differences.
   save('shadow_input_difference',wrapper-native)
   obs=native.clone();refs=jt.clone();repeat=target[:,to_target,None].transpose(1,2).expand(-1,5,-1)
   obs[1,1:7]=measured[1];refs[2]=repeat[2];obs[3,7:19]=wrist[3]
   for i in [4,6,7]:obs[i]=wrapper[i];refs[i]=repeat[i]
   obs[5,6]=5.;obs[6,6]=30.;obs[7,1:7]=native[7,1:7]
   td['action']=low(obs,refs)
   priv=(td['priv']-s['env']['transforms.4.loc'].to(env.device))/s['env']['transforms.4.scale'].to(env.device)
   teacher=TensorDict({'policy':(native-low.policy_loc)/low.policy_scale,'joint_target':(jt.flatten(1)-low.joint_loc)/low.joint_scale,'priv':priv},[11])
   agent.encoder_priv(teacher);agent.actor_teacher(teacher);td['action'][8]=teacher['action'][8]
   teacher['priv_pred']=teacher['priv_feature'];agent.actor_student(teacher);td['action'][9]=teacher['action'][9]
   agent.adapt_module(teacher);teacher['priv_feature']=teacher['priv_pred'];agent.actor_teacher(teacher);td['action'][10]=teacher['action'][10]
   save('action',td['action']);save('native_input',native);save('used_input',obs)
   td=env.step(td)['next'];assert cmd.force_apply_buffer.count_nonzero()==0
 arrays={k:np.stack(v) for k,v in rec.items()};np.savez_compressed(a.output/'traces.npz',**arrays)
 supervised=[robot.joint_names[i] for i in cmd.joint_idx_asset];sidx=[JOINT_NAMES.index(n) for n in supervised];omitted=[i for i in range(29) if i not in sidx]
 cases=[]
 for i,name in enumerate(labels):
  err=abs(arrays['q'][:,i]-arrays['target_q'][:,i]);root=np.linalg.norm(arrays['root_pos'][:,i]-arrays['target_root_pos'][:,i],axis=-1)
  cases.append(dict(case=name,joint_mae=float(err.mean()),supervised_joint_mae=float(err[:,sidx].mean()),omitted_joint_mae=float(err[:,omitted].mean()),per_joint_mae=err.mean(0).tolist(),root_position_error_m=float(root.mean()),first_fall_step=int(falls[i])))
 slices={'boot':[0,1],'root':[1,7],'wrists':[7,19],'angvel':[19,22],'gravity':[22,25],'q_history':[25,170],'prev_action':[170,257]}
 dif={name:float(abs(arrays['shadow_input_difference'][25:,0,lo:hi]).mean()) for name,(lo,hi) in slices.items()}
 result=dict(checkpoint=str(a.checkpoint.resolve()),clip=a.clip,frames=nsteps+1,network_equivalence_max_abs=equiv,supervised_joints=supervised,omitted_joints=[JOINT_NAMES[i] for i in omitted],joint_names=JOINT_NAMES,shadow_input_mae_after_boot=dif,cases=cases)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:sys.stdout.flush();sys.stderr.flush();os._exit(0)
