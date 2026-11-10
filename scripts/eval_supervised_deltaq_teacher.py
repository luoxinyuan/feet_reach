"""Short Isaac Sim smoke rollout for the supervised force->delta-q teacher."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import sys, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))

p=argparse.ArgumentParser()
p.add_argument('--ppo-checkpoint',type=Path,required=True,
               help='Checkpoint whose config supplies the CEER low-level simulator setup')
p.add_argument('--residual-checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--force-N',type=float,default=10.)
p.add_argument('--steps',type=int,default=600)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

ppo_state=torch.load(a.ppo_checkpoint,map_location='cpu',weights_only=False)
cfg=OmegaConf.create(OmegaConf.to_container(ppo_state['cfg'],resolve=True))
OmegaConf.set_struct(cfg,False); OmegaConf.set_readonly(cfg,False)
cfg.checkpoint_path=str(a.ppo_checkpoint.resolve()); cfg.vecnorm='eval'; cfg.seed=20261104
cfg.task.num_envs=1; cfg.task.max_episode_length=max(a.steps+10,610)
cfg.task.reward={}; cfg.task.termination={}; cfg.task.randomization={}
cfg.task.command.guidance_enabled=False; cfg.task.viewer.resolution=[960,540]
cfg.eval_render=True; cfg.headless=True; cfg.wandb.mode='disabled'
del ppo_state

app=AppLauncher(headless=True,device='cuda:0',enable_cameras=True,
                kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/omni.tbb.globalcontrol/maxThreadCount=8').app
try:
    from scripts.utils.helpers import make_env_policy
    from scripts.train_force_to_deltaq_residual import build_model
    from scripts.compliance_eval.visual import Video
    from scripts.compliance_eval.protocol import envelope
    from active_adaptation.utils.math import quat_apply
    from torchrl.envs.utils import set_exploration_type,ExplorationType
    from tensordict import TensorDict
    from torchrl.envs.transforms import VecNorm
    from active_adaptation.learning.hierarchical.dense_force_ppo import DenseForcePPO

    # Older dense-teacher checkpoints may contain auxiliary normalizers whose
    # dimensions differ from the current environment; the deployed environment
    # only needs compatible policy statistics for this simulator smoke rollout.
    def load_compatible_vecnorm_state(self,state):
        if state is None: raise KeyError('Missing VecNorm state in checkpoint')
        saved=TensorDict(state).flatten_keys(self.SEP)
        current=self._td.flatten_keys(self.SEP) if self._td is not None else TensorDict({},batch_size=[])
        compatible={}
        for key in saved.keys():
            value=saved.get(key); existing=current.get(key,None)
            if existing is None or existing.shape!=value.shape:
                if key in ('policy_sum','policy_ssq','policy_count'):
                    raise RuntimeError(f'Policy VecNorm mismatch for {key}: saved={tuple(value.shape)}, env={None if existing is None else tuple(existing.shape)}')
                continue
            compatible[key]=value
        current.update_(TensorDict(compatible,batch_size=[])); self._td=current.unflatten_keys(self.SEP)
    VecNorm.set_extra_state=load_compatible_vecnorm_state
    # Keep the checkpoint's deployed actor exact; only the unused privileged
    # critic may have a changed input width in today's environment config.
    original_dense_load=DenseForcePPO.load_state_dict
    def load_dense_for_smoke(self,state_dict,strict=True):
        state_dict=dict(state_dict); raw_teacher=state_dict.get('actor')
        actor=self.actor.module if hasattr(self.actor,'module') else self.actor
        saved_actor=state_dict.get('actor',{}); target=actor.state_dict()
        stripped={k[len('module.'):]:v for k,v in saved_actor.items() if k.startswith('module.')}
        if len(set(saved_actor)&set(target))<len(set(stripped)&set(target)):
            saved_actor={**{k:v for k,v in saved_actor.items() if not k.startswith('module.')},**stripped}
            state_dict['actor']=saved_actor
        mismatch=[k for k,v in target.items() if k not in saved_actor or saved_actor[k].shape!=v.shape]
        extra=sorted(set(saved_actor)-set(target))
        if mismatch or extra: raise RuntimeError(f'Actor checkpoint mismatch: missing={mismatch}, extra={extra}')
        critic=self.critic.module if hasattr(self.critic,'module') else self.critic
        saved_critic=state_dict.get('critic',{})
        state_dict['critic']={k:v for k,v in saved_critic.items() if k in critic.state_dict() and critic.state_dict()[k].shape==v.shape}
        original_teacher_load=self.teacher.load_state_dict
        if raw_teacher is not None:
            self.teacher.load_state_dict=lambda _weights,*args,**kwargs: original_teacher_load(raw_teacher,*args,**kwargs)
        try: return original_dense_load(self,state_dict,strict=False)
        finally: self.teacher.load_state_dict=original_teacher_load
    DenseForcePPO.load_state_dict=load_dense_for_smoke

    blob=torch.load(a.residual_checkpoint,map_location='cpu',weights_only=False)
    model=build_model().to('cuda:0').eval(); model.load_state_dict(blob['model'],strict=True)
    if len(blob['point_names'])!=3 or len(blob['joint_names'])!=29: raise RuntimeError('Unexpected model I/O schema')
    input_mean=torch.as_tensor(blob['input_mean_N'],device='cuda:0',dtype=torch.float32)
    input_std=torch.as_tensor(blob['input_std_N'],device='cuda:0',dtype=torch.float32)
    output_scale=torch.as_tensor(blob['output_scale_rad'],device='cuda:0',dtype=torch.float32)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    env,agent,_,_=make_env_policy(cfg); env.eval(); env.base_env.eval(); agent.eval()
    cmd=env.command_manager; robot=cmd.asset; manager=env.action_manager; device=env.device
    if list(cmd.point_names)!=list(blob['point_names']): raise RuntimeError(f"Force point mismatch: {cmd.point_names}")
    if list(manager.joint_names)!=list(blob['joint_names']): raise RuntimeError('delta-q and CEER action joint orders differ')
    cmd.zero_init_prob=1.
    for key in cmd.init_noise_params: cmd.init_noise_params[key]=0.
    original_reset=cmd.reset
    def standing(ids):
        original_reset(ids)
        root=robot.data.default_root_state[ids].clone(); root[:,:3]+=env.scene.env_origins[ids]
        root[:,3:7]=root.new_tensor([1.,0,0,0]); root[:,7:]=0
        q=robot.data.default_joint_pos[ids].clone()
        robot.write_root_state_to_sim(root,env_ids=ids)
        robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        cmd.joint_pos_boot_protect[ids]=q
        for key in ('force','desired','force_apply_buffer','torque_apply_buffer'):
            if hasattr(cmd,key): getattr(cmd,key)[ids]=0
    callbacks=env.base_env._reset_callbacks
    callbacks[:]=[standing if cb==original_reset else cb for cb in callbacks]
    if standing not in callbacks: raise RuntimeError('Failed to install deterministic initial state')
    cmd.before_update=lambda:setattr(cmd,'log_step',cmd.log_step+1)
    cmd.update=lambda:None

    # Same behavior-eval stimulus: 10 N downward at the pelvis, ramped 2-3 s,
    # held through 7 s, then released. The video draws the actual applied force.
    n=1; body_ids=torch.zeros((1,2),dtype=torch.long,device=device)
    body_ids[0,0]=robot.body_names.index('pelvis')
    vectors=torch.zeros((1,2,3),device=device); vectors[0,0,2]=-float(a.force_N)
    offsets=torch.zeros_like(vectors); applied=torch.zeros_like(vectors)
    def force_step(substep):
        t=clock_step*env.step_dt+substep*env.physics_dt
        applied.copy_(vectors*envelope(t))
        cmd.force_apply_buffer.zero_(); cmd.torque_apply_buffer.zero_()
        cmd.force_apply_buffer.scatter_add_(1,body_ids[...,None].expand_as(applied),applied)
        if hasattr(cmd,'point_offsets'):
            cmd.point_force_buffer.zero_()
            for pi,bid in enumerate(cmd.point_ids.tolist()):
                if float(cmd.point_offsets[pi].norm())<1e-8:
                    cmd.point_force_buffer[:,pi]=((body_ids==bid)[...,None]*applied).sum(1)
        cmd.position_apply_buffer.copy_(robot.data.body_pos_w)
        robot.has_external_wrench=False
    clock_step=0
    original_step=cmd.step; pre=env.base_env._pre_step_callbacks
    pre[:]=[force_step if cb==original_step else cb for cb in pre]
    if force_step not in pre: raise RuntimeError('Failed to install force stimulus')

    td=env.reset(); env.base_env.episode_length_buf.zero_()
    low=getattr(manager,'low_action_manager',manager)
    if hasattr(low,'alpha_jit_scale'): low.alpha_jit_scale=None
    if hasattr(low,'alpha'): low.alpha.fill_(sum(low.alpha_range)/2)
    if hasattr(low,'delay'): low.delay.fill_((low.min_delay+low.max_delay)//2)
    if hasattr(low,'boot_delay'): low.boot_delay.copy_(low.delay)
    renderer=Video(env,cmd,a.output/'rollout.mp4',False)
    dt=float(env.step_dt); speed=float(manager.reference.speed)
    rows=[]; first_fall=None; max_action_saturation=0.; pred_norms=[]; force_norms=[]
    with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
        for step in range(a.steps+1):
            t=step*dt
            root=robot.data.root_pos_w[0]-env.scene.env_origins[0]
            tilt=torch.acos((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))
            q=robot.data.joint_pos[0]
            fallen=bool(root[2]<.30 or tilt>np.deg2rad(75) or not torch.isfinite(q).all())
            if fallen and first_fall is None: first_fall=t
            forces=cmd.force_apply_buffer[0]
            rows.append({'time_s':t,'root_z_m':float(root[2]),'tilt_deg':float(torch.rad2deg(tilt)),
                         'pelvis_force_N':float(forces[robot.body_names.index('pelvis')].norm()),
                         'predicted_delta_q_rms_rad':pred_norms[-1] if pred_norms else 0.})
            if step%2==0 or step==a.steps:
                renderer.frame(t,'supervised Δq teacher • pelvis crouch prompt',fallen)
            if step==a.steps: break
            clock_step=step
            # dense_force_gt is normalized by 30 N and expressed in root coordinates.
            f=(td['force_gt_'].reshape(1,3,3)*30.0)[0].float()
            pred=model(((f.reshape(1,9)-input_mean)/input_std))*output_scale
            dq=pred[0]
            # CEERResidualAction integrates tanh(action)*dt*speed into its 29-joint
            # reference. Convert supervised physical Δq (rad/control step) to that
            # interface, clipping only where the reference integrator saturates.
            ratio=(dq/(dt*speed)).clamp(-.95,.95)
            raw=torch.atanh(ratio).unsqueeze(0)
            max_action_saturation=max(max_action_saturation,float((dq.abs()>dt*speed*.95).float().mean()))
            pred_norms.append(float(dq.square().mean().sqrt()))
            force_norms.append(float(f.norm()))
            td['action']=raw
            td=env.step(td)['next']
    renderer.close()
    import csv
    with (a.output/'trace.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={'residual_checkpoint':str(a.residual_checkpoint.resolve()),'base_sim_config_checkpoint':str(a.ppo_checkpoint.resolve()),
            'stimulus':'10 N downward pelvis force; eval envelope 2-3 s ramp, 3-7 s hold, 7-8 s release',
            'control_adapter':'supervised next-step Δq -> CEERResidualAction reference increment; raw=atanh(Δq/(dt*reference_speed))',
            'duration_s':a.steps*dt,'steps':a.steps,'dt_s':dt,'force_points':list(blob['point_names']),
            'max_predicted_delta_q_rms_rad':max(pred_norms,default=0.),'mean_predicted_delta_q_rms_rad':float(np.mean(pred_norms)) if pred_norms else 0.,
            'mean_force_input_norm_N':float(np.mean(force_norms)) if force_norms else 0.,
            'fraction_saturated_joint_steps_max':max_action_saturation,'first_fall_s':first_fall,
            'root_z_min_m':float(min(r['root_z_m'] for r in rows)),'video':'rollout.mp4','trace':'trace.csv'}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUPERVISED_DELTAQ_EVAL_COMPLETE '+json.dumps(report),flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle=True
    env.close()
except BaseException:
    import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    app.close()
