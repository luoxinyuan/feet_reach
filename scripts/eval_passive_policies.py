"""Deterministic held-out first-episode evaluation; no resets hide failures."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['compliance','locomotion','whole-body'],required=True)
 p.add_argument('--controller',choices=['teacher','student','adapt'],default='student')
 p.add_argument('--scenario',choices=['guided','zero','hold'],default='guided')
 p.add_argument('--num-envs',type=int,default=256);p.add_argument('--steps',type=int,default=500)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--video',action='store_true')
 p.add_argument('--checkpoint',type=Path,help='Direct residual checkpoint; evaluates public-only actor')
 p.add_argument('--run-dir',type=Path,help='Override experiment directory, e.g. an AMASS EE run')
 p.add_argument('--dataset-path',help='Override whole-body MotionDataset mem_path')
 p.add_argument('--guidance-cache',type=Path,help='Override whole-body segmented force cache')
 p.add_argument('--postforce-window-steps',type=int,default=25,
                help='Early post-force-off window in control steps (default 25 = 0.5 s at 50 Hz)')
 p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 import torch
 from omegaconf import OmegaConf
 import active_adaptation.learning
 from isaaclab.app import AppLauncher
 run=args.run_dir or (args.checkpoint.parent if args.checkpoint else None) or Path((ROOT/f'outputs/passive-{args.task}/latest_pipeline.txt').read_text().strip())
 phase={'teacher':'train','adapt':'adapt','student':'finetune'}[args.controller]
 checkpoint=args.checkpoint or next((run/phase).rglob('checkpoint_final.pt'))
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
 cfg.checkpoint_path=str(checkpoint);cfg.vecnorm='eval';cfg.seed=args.seed;cfg.task.num_envs=args.num_envs
 cfg.wandb.mode='disabled';cfg.eval_render=args.video;cfg.headless=True
 continuous_ee=cfg.task.command._target_.split('.')[-1] in {'PassiveEEContinuousCommand','PassiveEEDenseCommand','PassiveEECEERCommand'}
 if args.task=='compliance':
  if continuous_ee:
   cfg.task.command.dataset.mem_paths=[path.replace('/train/','/val/') for path in cfg.task.command.dataset.mem_paths]
   cfg.task.command.dataset.path_weights=[0.,1.] if args.scenario=='hold' else [1.,0.]
  else:
   cfg.task.command.dataset.mem_paths=[str(Path(path).parent/'val') for path in cfg.task.command.dataset.mem_paths]
   if args.scenario=='hold':cfg.task.command.hold_probability=1.
 elif args.task=='whole-body':
  if args.scenario=='hold':raise ValueError('Full dataset has no separate frozen hold split')
  if args.dataset_path:
   cfg.task.command.dataset.mem_paths=[args.dataset_path]
   cfg.task.command.dataset.path_weights=[1.]
  else:
   cfg.task.command.dataset.mem_paths=['passive_whole_body_full/val']
   cfg.task.command.dataset.path_weights=[1.]
  if 'guidance_cache' in cfg.task.command:
   cfg.task.command.guidance_cache=str(args.guidance_cache.resolve()) if args.guidance_cache else str(ROOT/'artifacts/ceer_segmented_jacobian/val18')
 else:
  cfg.task.command.dataset.mem_paths=['passive_locomotion_simple/val/move','passive_locomotion_simple/val/hold']
  cfg.task.command.dataset.path_weights=[0.,1.] if args.scenario=='hold' else [1.,0.]
 cfg.task.viewer.eye=[3.,3.,2.5];cfg.task.viewer.lookat=[0.,0.,.8]
 del state
 app=AppLauncher(headless=True,device='cuda:0',enable_cameras=args.video).app
 try:
  from scripts.utils.helpers import make_env_policy
  from torchrl.envs.utils import set_exploration_type,ExplorationType
  from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
  from active_adaptation.utils.math import quat_apply_inverse
  torch.manual_seed(args.seed);np.random.seed(args.seed)
  env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval()
  cmd=env.command_manager;asset=cmd.asset
  single_hand_side_by_motion=None
  if args.task=='whole-body' and args.dataset_path and args.guidance_cache:
   reports=json.loads((args.guidance_cache/'manifest.json').read_text())['reports']
   side_ids=torch.full((len(reports),),-1,dtype=torch.long,device=env.device)
   for r in reports:side_ids[int(r['motion_id'])]=0 if r['side']=='left' else 1
   single_hand_side_by_motion=side_ids
   left_hand_asset=asset.body_names.index('left_hand_mimic');right_hand_asset=asset.body_names.index('right_hand_mimic')
   left_hand_motion=cmd.dataset.body_names.index('left_hand_mimic');right_hand_motion=cmd.dataset.body_names.index('right_hand_mimic')
  if args.checkpoint:cmd.zero_init_prob=1.
  if args.task=='compliance':cmd.distance_max=.08
  if args.scenario=='hold':cmd.zero_init_prob=1.
  if args.scenario=='zero':
   # Patch the registered physics callback, including its public force observation.
   original=cmd.step
   def zero_step(substep):
    original(substep);cmd.force_apply_buffer.zero_()
    if args.task=='compliance':cmd.ee_force.zero_()
    elif args.task=='whole-body':
     cmd.torque_apply_buffer.zero_();cmd.force.zero_();cmd.desired.zero_()
     if hasattr(cmd,'point_force_buffer'):cmd.point_force_buffer.zero_()
    else:cmd.torque_apply_buffer.zero_();cmd.applied_force.zero_();cmd.applied_torque.zero_()
   callbacks=env.base_env._pre_step_callbacks
   callbacks[:]=[zero_step if callback==original else callback for callback in callbacks]
   assert zero_step in callbacks
  policy=agent.get_rollout_policy('train' if args.controller=='teacher' else 'deploy')
  if args.controller!='teacher':assert list(policy.in_keys)==['policy']
  torch.manual_seed(args.seed);carry=env.reset();n=args.num_envs;device=env.device
  episode_side=(single_hand_side_by_motion[cmd._motion.motion_id[:,0].long()]
                if single_hand_side_by_motion is not None else None)
  root0=asset.data.root_pos_w.clone();yaw0=yaw_from_quat(asset.data.root_quat_w).clone()
  initial=asset.data.joint_pos.detach().cpu().numpy().tobytes()+root0.cpu().numpy().tobytes()
  alive=torch.ones(n,device=device,dtype=torch.bool);failed=~alive;fallen=~alive;completed=~alive
  lengths=torch.zeros(n,device=device);sums={};count=0.;max_drift=torch.zeros(n,device=device)
  if single_hand_side_by_motion is not None:
   passive_lift_max=torch.zeros(n,device=device);side_seen=torch.zeros(n,dtype=torch.bool,device=device)
   force_seen=torch.zeros(n,dtype=torch.bool,device=device);force_off_seen=torch.zeros(n,dtype=torch.bool,device=device)
   force_off_step=torch.full((n,),-1,dtype=torch.long,device=device)
   passive_at_force_off=torch.zeros(n,3,device=device);active_at_force_off=torch.zeros(n,3,device=device)
   postforce_passive_excursion=torch.zeros(n,device=device);postforce_active_excursion=torch.zeros(n,device=device)
   postforce_final_passive_displacement=torch.zeros(n,device=device);postforce_final_active_displacement=torch.zeros(n,device=device)
   postforce_sums={s:{k:0. for k in ('passive_error_at_force_off_m','passive_error_first_window_m',
       'passive_error_postforce_m','passive_z_abs_error_postforce_m','postforce_samples',
       'first_window_samples')} for s in ('left','right')}
   postforce_episodes={s:0 for s in ('left','right')}
   side_steps={s:0. for s in ('left','right')};side_sums={s:{'non_forced_hand_upward_error_m':0.,'non_forced_hand_abs_z_error_m':0.,'non_forced_hand_error_m':0.,'active_hand_error_m':0.,'lift_gt_5cm_steps':0.,'lift_gt_10cm_steps':0.} for s in ('left','right')}
  trace=[];frames=[];names=asset.body_names
  with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
   for step in range(args.steps):
    carry=policy(carry);td=env.step(carry);carry=td['next']
    qerr=(asset.data.joint_pos[:,cmd.joint_idx_asset]-cmd._motion.joint_pos[:,0,cmd.joint_idx_motion]).abs().mean(-1)
    rooterr=(asset.data.root_pos_w[:,:2]-cmd.reward_root_pos_w[:,:2]).norm(dim=-1)
    yawerr=yaw_from_quat(asset.data.root_quat_w)-yaw_from_quat(cmd.reward_root_quat_w)
    yawerr=torch.atan2(yawerr.sin(),yawerr.cos()).abs()
    drift=(asset.data.root_pos_w[:,:2]-root0[:,:2]).norm(dim=-1)
    metrics={'joint_mae_rad':qerr,'root_xy_error_m':rooterr,'yaw_error_rad':yawerr,
      'root_speed_m_s':asset.data.root_lin_vel_w[:,:2].norm(dim=-1),'root_drift_m':drift}
    if args.task=='compliance':
     ee_target=(cmd._motion.body_pos_w[:,0,cmd.motion_hand_ids]+env.scene.env_origins[:,None]) if continuous_ee else cmd.anchor
     err=(asset.data.body_pos_w[:,cmd.hand_ids]-ee_target).norm(dim=-1)
     metrics.update(ee_error_m=err.mean(-1),ee_within_5cm=(err.amax(-1)<.05).float(),force_N=cmd.ee_force.norm(dim=-1).mean(-1))
     if continuous_ee:
      actual_b=quat_apply_inverse(asset.data.root_quat_w[:,None].expand(-1,2,-1),asset.data.body_pos_w[:,cmd.hand_ids]-asset.data.root_pos_w[:,None])
      target_b=quat_apply_inverse(cmd._motion.root_quat_w[:,0,None].expand(-1,2,-1),cmd._motion.body_pos_w[:,0,cmd.motion_hand_ids]-cmd._motion.root_pos_w[:,0,None])
      metrics['ee_root_relative_error_m']=(actual_b-target_b).norm(dim=-1).mean(-1)
      qdiff=(asset.data.joint_pos[:,cmd.joint_idx_asset]-cmd._motion.joint_pos[:,0,cmd.joint_idx_motion]).abs()
      arm_mask=[j for j,i in enumerate(cmd.joint_idx_asset.tolist()) if any(s in asset.joint_names[i] for s in ('shoulder','elbow','wrist'))]
      metrics['arm_joint_mae_rad']=qdiff[:,arm_mask].mean(-1)
      for j,i in enumerate(cmd.joint_idx_asset.tolist()):
       metrics['joint_mae_rad/'+asset.joint_names[i]]=qdiff[:,j]
    elif args.task=='whole-body':
     actual=asset.data.body_pos_w[:,cmd.keypoint_idx_asset]
     target=cmd.reward_keypoints_w[:,cmd.keypoint_idx_motion]
     forces=cmd.force_apply_buffer;arms=asset.data.body_pos_w-asset.data.root_pos_w[:,None]
     net=forces.sum(1);yaw=(torch.cross(arms,forces,dim=-1)+cmd.torque_apply_buffer).sum(1)[:,2]
     point_forces=cmd.point_force_buffer if hasattr(cmd,'point_force_buffer') else forces
     force_magnitude=point_forces.norm(dim=-1).sum(-1)
     if hasattr(cmd,'point_desired'):
      desired_force=cmd.point_desired.sum(1)
      desired_yaw=torch.cross(cmd.point_positions_w()-asset.data.root_pos_w[:,None],cmd.point_desired,dim=-1).sum(1)[:,2]
     else:desired_force,desired_yaw=cmd.desired_force,cmd.desired_yaw
     metrics.update(keypoint_error_m=(actual-target).norm(dim=-1).mean(-1),
       force_N=force_magnitude,active_force_count=(point_forces.norm(dim=-1)>1e-5).sum(-1).float(),
       force_residual_N=(net-desired_force).norm(dim=-1),yaw_residual_Nm=(yaw-desired_yaw).abs(),
       yaw_intent_agree=((yaw*desired_yaw)>=0).float())
     qdiff=(asset.data.joint_pos[:,cmd.joint_idx_asset]-cmd._motion.joint_pos[:,0,cmd.joint_idx_motion]).abs()
     for j,i in enumerate(cmd.joint_idx_asset.tolist()):metrics['joint_mae_rad/'+asset.joint_names[i]]=qdiff[:,j]
     if single_hand_side_by_motion is not None:
      side=episode_side
      if bool((side<0).any()):raise RuntimeError('Sampled motion has no active-hand label in eval manifest')
      # Dataset motion poses are stored in the environment-local frame, while
      # articulation body poses are world-frame and include the tiled env offset.
      hands=(asset.data.body_pos_w[:,[left_hand_asset,right_hand_asset]]
             -env.scene.env_origins[:,None,:])
      refs=cmd._motion.body_pos_w[:,0,[left_hand_motion,right_hand_motion]]
      active_left=side==0
      env_idx=torch.arange(n,device=device)
      active_i=torch.where(active_left,0,1);passive_i=1-active_i
      active_err=(hands[env_idx,active_i]-refs[env_idx,active_i]).norm(dim=-1)
      passive_delta_z=hands[env_idx,passive_i,2]-refs[env_idx,passive_i,2]
      passive_error=(hands[env_idx,passive_i]-refs[env_idx,passive_i]).norm(dim=-1)
      upward=passive_delta_z.clamp_min(0.)
      force_active=force_magnitude>1e-5
      force_off_now=force_seen&(~force_active)&(~force_off_seen)
      force_seen|=force_active
      force_off_step=torch.where(force_off_now,torch.full_like(force_off_step,step),force_off_step)
      force_off_seen|=force_off_now
      passive_xyz=hands[env_idx,passive_i]
      active_xyz=hands[env_idx,active_i]
      passive_at_force_off=torch.where(force_off_now[:,None],passive_xyz,passive_at_force_off)
      active_at_force_off=torch.where(force_off_now[:,None],active_xyz,active_at_force_off)
      postforce_mask=alive&force_off_seen&(force_off_step>=0)&(~force_active)
      if postforce_mask.any():
       passive_from_off=(passive_xyz-passive_at_force_off).norm(dim=-1)
       active_from_off=(active_xyz-active_at_force_off).norm(dim=-1)
       postforce_passive_excursion=torch.where(postforce_mask,torch.maximum(postforce_passive_excursion,passive_from_off),postforce_passive_excursion)
       postforce_active_excursion=torch.where(postforce_mask,torch.maximum(postforce_active_excursion,active_from_off),postforce_active_excursion)
       postforce_final_passive_displacement=torch.where(postforce_mask,passive_from_off,postforce_final_passive_displacement)
       postforce_final_active_displacement=torch.where(postforce_mask,active_from_off,postforce_final_active_displacement)
       age=step-force_off_step
       for si,sname in enumerate(('left','right')):
        sample_mask=postforce_mask&(side==si)
        postforce_sums[sname]['passive_error_postforce_m']+=passive_error[sample_mask].sum().item()
        postforce_sums[sname]['passive_z_abs_error_postforce_m']+=passive_delta_z[sample_mask].abs().sum().item()
        postforce_sums[sname]['postforce_samples']+=int(sample_mask.sum().item())
        window_mask=sample_mask&(age<args.postforce_window_steps)
        postforce_sums[sname]['passive_error_first_window_m']+=passive_error[window_mask].sum().item()
        postforce_sums[sname]['first_window_samples']+=int(window_mask.sum().item())
      for si,sname in enumerate(('left','right')):
       off_mask=alive&force_off_now&(side==si)
       postforce_sums[sname]['passive_error_at_force_off_m']+=passive_error[off_mask].sum().item()
       postforce_episodes[sname]+=int(off_mask.sum().item())
      metrics.update(active_hand_tracking_error_m=active_err,
       non_forced_hand_upward_error_m=upward,
       non_forced_hand_abs_z_error_m=passive_delta_z.abs(),
       non_forced_hand_error_m=passive_error,
       non_forced_hand_lift_gt_5cm=(passive_delta_z>.05).float(),
       non_forced_hand_lift_gt_10cm=(passive_delta_z>.10).float())
      passive_lift_max=torch.maximum(passive_lift_max,upward)
      side_seen|=alive
      for si,sname in enumerate(('left','right')):
       mask=alive&(side==si);side_steps[sname]+=int(mask.sum().item())
       for key,val in [('non_forced_hand_upward_error_m',upward),('non_forced_hand_abs_z_error_m',passive_delta_z.abs()),('non_forced_hand_error_m',passive_error),('active_hand_error_m',active_err)]:side_sums[sname][key]+=val[mask].sum().item()
       side_sums[sname]['lift_gt_5cm_steps']+=(passive_delta_z[mask]>.05).sum().item()
       side_sums[sname]['lift_gt_10cm_steps']+=(passive_delta_z[mask]>.10).sum().item()
    else:metrics.update(force_N=cmd.applied_force.norm(dim=-1),yaw_torque_Nm=cmd.applied_torque[:,2].abs())
    for k,v in metrics.items():sums[k]=sums.get(k,0.)+v[alive].sum().item()
    count+=alive.sum().item();lengths+=alive.float();max_drift=torch.where(alive,torch.maximum(max_drift,drift),max_drift)
    fall=(asset.data.root_pos_w[:,2]-env.scene.env_origins[:,2]<.45)|(asset.data.projected_gravity_b[:,2]>-.5)
    terminated=carry['terminated'].flatten();truncated=carry['truncated'].flatten()
    failed|=alive&terminated;fallen|=alive&fall;completed|=alive&truncated&~terminated&~fall
    trace.append(dict(root=(asset.data.root_pos_w[:8]-env.scene.env_origins[:8]).cpu().numpy(),
      target=(cmd.reward_root_pos_w[:8]-env.scene.env_origins[:8]).cpu().numpy(),
      bodies=(asset.data.body_pos_w[:8]-env.scene.env_origins[:8,None]).cpu().numpy(),
      alive=alive[:8].cpu().numpy(),joint_error=qerr[:8].cpu().numpy()))
    if args.video and step%2==0:frames.append(env.render('rgb_array').copy())
    alive&=~(terminated|truncated|fall)
    if not alive.any():break
  completed|=alive
  report=dict(task=args.task,controller=args.controller,scenario=args.scenario,seed=args.seed,
   num_envs=n,checkpoint=str(checkpoint),initial_sha256=hashlib.sha256(initial).hexdigest(),
   dataset=list(cfg.task.command.dataset.mem_paths),deterministic=True,
   reference_sha256=hashlib.sha256(cmd.dataset._buf_A.joint_pos.cpu().numpy().tobytes()).hexdigest() if args.checkpoint else None,
   mean_duration_s=lengths.mean().item()*env.step_dt,completion_fraction=completed.float().mean().item(),
   task_termination_fraction=failed.float().mean().item(),physical_fall_fraction=fallen.float().mean().item(),
   max_root_drift_mean_m=max_drift.mean().item(),max_root_drift_p95_m=max_drift.quantile(.95).item(),
   sample_weighted_metrics={k:v/count for k,v in sums.items()},
   notes='First episode only; stop counting at task termination, clip end, physical fall (height<0.45m or tilt>60deg), or horizon. Completion includes clip end; time-weighted errors include failure step.')
  if single_hand_side_by_motion is not None:
   by_side={}
   for si,sname in enumerate(('left','right')):
    env_mask=side_seen&(episode_side==si)
    per_env=passive_lift_max[env_mask]
    den=max(side_steps[sname],1)
    by_side[sname]=dict(episodes=int(env_mask.sum().item()),time_samples=side_steps[sname],
     mean_non_forced_hand_peak_upward_error_m=per_env.mean().item() if len(per_env) else None,
     p95_non_forced_hand_peak_upward_error_m=per_env.quantile(.95).item() if len(per_env) else None,
     episodes_peak_lift_gt_5cm=float((per_env>.05).float().mean().item()) if len(per_env) else None,
     episodes_peak_lift_gt_10cm=float((per_env>.10).float().mean().item()) if len(per_env) else None,
     mean_non_forced_hand_upward_error_m=side_sums[sname]['non_forced_hand_upward_error_m']/den,
     mean_non_forced_hand_abs_z_error_m=side_sums[sname]['non_forced_hand_abs_z_error_m']/den,
     mean_non_forced_hand_3d_error_m=side_sums[sname]['non_forced_hand_error_m']/den,
     mean_active_hand_3d_error_m=side_sums[sname]['active_hand_error_m']/den,
     time_fraction_non_forced_hand_lift_gt_5cm=side_sums[sname]['lift_gt_5cm_steps']/den,
     time_fraction_non_forced_hand_lift_gt_10cm=side_sums[sname]['lift_gt_10cm_steps']/den)
   report['single_hand_metrics_by_active_side']=by_side
   postforce_by_side={}
   for sname in ('left','right'):
    si=0 if sname=='left' else 1
    env_mask=side_seen&(episode_side==si)&force_off_seen
    samples=max(postforce_sums[sname]['postforce_samples'],1)
    window_samples=max(postforce_sums[sname]['first_window_samples'],1)
    episode_count=max(postforce_episodes[sname],1)
    postforce_by_side[sname]=dict(episodes_force_ended=int(env_mask.sum().item()),
      mean_passive_3d_error_at_force_off_m=postforce_sums[sname]['passive_error_at_force_off_m']/episode_count,
      mean_passive_3d_error_first_window_m=postforce_sums[sname]['passive_error_first_window_m']/window_samples,
      mean_passive_3d_error_after_force_off_m=postforce_sums[sname]['passive_error_postforce_m']/samples,
      mean_passive_abs_vertical_error_after_force_off_m=postforce_sums[sname]['passive_z_abs_error_postforce_m']/samples,
      mean_passive_excursion_force_off_to_episode_end_m=postforce_passive_excursion[env_mask].mean().item() if env_mask.any() else None,
      mean_passive_displacement_force_off_to_episode_end_m=postforce_final_passive_displacement[env_mask].mean().item() if env_mask.any() else None,
      mean_active_displacement_force_off_to_episode_end_m=postforce_final_active_displacement[env_mask].mean().item() if env_mask.any() else None)
   report['post_force_off_metrics_by_active_side']=dict(window_seconds=args.postforce_window_steps*env.step_dt,
       metrics=postforce_by_side)
  (args.output/'summary.json').write_text(json.dumps(report,indent=2))
  np.savez_compressed(args.output/'traces.npz',**{k:np.stack([t[k] for t in trace]) for k in trace[0]},body_names=np.array(names),step_dt=env.step_dt)
  if frames:
   import imageio.v2 as imageio
   imageio.mimwrite(args.output/'rollout.mp4',frames,fps=round(1/env.step_dt/2))
  print('EVALUATION_COMPLETE',json.dumps(report),flush=True)
  from isaaclab.sim import SimulationContext
  SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
 except BaseException:
  import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
 else:app.close()
if __name__=='__main__':main()
