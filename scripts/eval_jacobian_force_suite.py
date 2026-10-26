"""Target-free hand/torso translation and shoulder-couple interaction trials."""
import os,sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1');os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--video-group',choices=['left','right','body','turn','sequence']);p.add_argument('--metrics-pass',choices=['forced','zero']);p.add_argument('--repeats',type=int,default=8);p.add_argument('--video-force',type=float,default=10.);p.add_argument('--basic-only',action='store_true');p.add_argument('--sequence-gap',type=float,default=2.);p.add_argument('--sequence-ee-force',type=float,default=5.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if not a.video_group and not a.metrics_pass:p.error('Use separate fresh processes with --metrics-pass forced and --metrics-pass zero')
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
DIRECTIONS={'forward':[1.,0.,0.],'backward':[-1.,0.,0.],'left':[0.,1.,0.],'right':[0.,-1.,0.]}
CASES=[dict(name=g+'_'+d,group=g,direction=d,body=b,vector=v,turn=0) for g,b in [('left','left_hand_mimic'),('right','right_hand_mimic'),('body','torso_link')] for d,v in DIRECTIONS.items()]
CASES += [dict(name='turn_'+d,group='turn',direction=d,body='left_shoulder_roll_link',vector=[0.,0.,0.],turn=s) for d,s in [('left',1),('right',-1)]]
CASES += [dict(name='shoulder_then_ee',group='sequence',direction='left',body='left_shoulder_roll_link',vector=[0.,0.,0.],turn=2)]
if a.basic_only:CASES=[c for c in CASES if c['group']!='sequence']
if a.video_group:
 cases=[x for x in CASES if x['group']==a.video_group];rows=[dict(c,force_N=a.video_force,repeat=0) for c in cases];n=1
else:
 rows=[dict(c,force_N=f,repeat=r) for c in CASES for f in [5.,10.,20.] for r in range(a.repeats)];n=len(rows)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
assert cfg.task.command._target_.split('.')[-1] in ('JacobianComplianceCommand','SegmentedJacobianComplianceCommand','AMPComplianceCommand')
cfg.checkpoint_path=str(a.checkpoint);cfg.vecnorm='eval';cfg.task.num_envs=n;cfg.seed=42;cfg.wandb.mode='disabled';cfg.eval_render=bool(a.video_group);cfg.headless=True
cfg.task.command.guidance_enabled=False;cfg.task.command.dataset.mem_paths=['passive_whole_body_full/val'];cfg.task.reward={};cfg.task.termination={};cfg.task.max_episode_length=2000;cfg.task.observe_on_reset=True;cfg.task.viewer.resolution=[1280,720]
if 'guidance_cache' in cfg.task.command:cfg.task.command.guidance_cache=str(ROOT/'artifacts/ceer_segmented_jacobian/val18')
del state
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=bool(a.video_group),kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/plugins/omni.tbb.globalcontrol/maxThreadCount=16').app
try:
 from scripts.utils.helpers import make_env_policy
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 from active_adaptation.utils.body_guidance import slew_force
 from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
 torch.manual_seed(42);np.random.seed(42)
 env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval();cmd=env.command_manager;robot=cmd.asset;manager=env.action_manager;device=env.device
 original=cmd.reset
 def reset_standing(ids):
  original(ids)
  root=robot.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids];root[:,3:7]=root.new_tensor([1.,0.,0.,0.]);root[:,7:]=0
  robot.write_root_state_to_sim(root,env_ids=ids);robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],torch.zeros_like(robot.data.default_joint_vel[ids]),env_ids=ids)
  cmd.joint_pos_boot_protect[ids]=robot.data.default_joint_pos[ids];cmd.force[ids]=0;cmd.desired[ids]=0;cmd.force_apply_buffer[ids]=0;cmd.torque_apply_buffer[ids]=0
  if hasattr(cmd,'point_force_buffer'):cmd.point_force_buffer[ids]=0
  # Seed AFTER dataset initialization, BEFORE actuator and observation resets.
  torch.manual_seed(4242)
 callbacks=env.base_env._reset_callbacks;callbacks[:]=[reset_standing if cb==original else cb for cb in callbacks];assert reset_standing in callbacks
 # Freeze dataset progression; it supplies neither prompts, rewards nor actor targets.
 cmd.before_update=lambda:None;cmd.update=lambda:None
 applied=torch.zeros(n,2,3,device=device);clock_step=0;force_enabled=True
 left=robot.body_names.index('left_shoulder_roll_link');right=robot.body_names.index('right_shoulder_roll_link')
 left_ee=robot.body_names.index('left_hand_mimic');right_ee=robot.body_names.index('right_hand_mimic')
 indices=torch.arange(n,device=device)[:,None]
 def configure(trials):
  global body_ids,vectors,turns,magnitudes
  body_ids=torch.tensor([[robot.body_names.index(c['body']),right if c['turn'] else robot.body_names.index(c['body'])] for c in trials],device=device)
  vectors=torch.tensor([c['vector'] for c in trials],device=device);turns=torch.tensor([c['turn'] for c in trials],device=device);magnitudes=torch.tensor([c['force_N'] for c in trials],device=device)
 def external_force(substep):
  t=(clock_step+substep/4)*env.step_dt
  amplitude=0. if t<=2 or t>=6 else .5-.5*np.cos(np.pi*(t-2)/.5) if t<2.5 else .5-.5*np.cos(np.pi*(6-t)/.5) if t>5.5 else 1.
  sequence=turns.abs()==2
  ee_start=5.+a.sequence_gap;ee_end=ee_start+3.
  shoulder_amp=0. if t<=2 or t>=5 else .5-.5*np.cos(np.pi*(t-2)/.5) if t<2.5 else .5-.5*np.cos(np.pi*(5-t)/.5) if t>4.5 else 1.
  ee_amp=0. if t<=ee_start or t>=ee_end else .5-.5*np.cos(np.pi*(t-ee_start)/.5) if t<ee_start+.5 else .5-.5*np.cos(np.pi*(ee_end-t)/.5) if t>ee_end-.5 else 1.
  desired=torch.zeros_like(applied);desired[:,0]=vectors*magnitudes[:,None]
  pair_left=torch.where(sequence,torch.full_like(turns,left_ee),torch.full_like(turns,left))
  pair_right=torch.where(sequence,torch.full_like(turns,right_ee),torch.full_like(turns,right))
  shoulder_phase=sequence & (t<ee_start)
  pair_left=torch.where(shoulder_phase,torch.full_like(turns,left),pair_left)
  pair_right=torch.where(shoulder_phase,torch.full_like(turns,right),pair_right)
  separation=robot.data.body_pos_w[indices[:,0],pair_left]-robot.data.body_pos_w[indices[:,0],pair_right];separation=separation.clone();separation[:,2]=0
  tangent=torch.stack((-separation[:,1],separation[:,0],torch.zeros(n,device=device)),-1)/separation.norm(dim=-1,keepdim=True).clamp_min(.05)
  signed_turn=torch.where(sequence,torch.ones_like(turns),turns)
  pair_magnitudes=magnitudes.clone()
  if sequence.any() and t>=ee_start:pair_magnitudes[sequence]=a.sequence_ee_force
  pair=tangent*(signed_turn*pair_magnitudes)[:,None];mask=turns!=0
  desired[mask,0]=pair[mask];desired[mask,1]=-pair[mask]
  desired*=amplitude if force_enabled else 0.
  if sequence.any():
   sequence_amplitude=shoulder_amp if t<ee_start else ee_amp
   desired[sequence,0]=pair[sequence]*sequence_amplitude if force_enabled else 0.
   desired[sequence,1]=-pair[sequence]*sequence_amplitude if force_enabled else 0.
   body_ids[sequence,0]=pair_left[sequence];body_ids[sequence,1]=pair_right[sequence]
  applied.copy_(slew_force(applied,desired,env.physics_dt));cmd.force_apply_buffer.zero_();cmd.torque_apply_buffer.zero_()
  cmd.force_apply_buffer.scatter_add_(1,body_ids[...,None].expand_as(applied),applied)
  if hasattr(cmd,'point_force_buffer'):
   # Public force features must receive the same forces as the simulator.
   # All scripted loads are at link origins, not offset candidates on that link.
   cmd.point_force_buffer.zero_()
   for pi,bid in enumerate(cmd.point_ids.tolist()):
    if float(cmd.point_offsets[pi].norm())<1e-8:
     cmd.point_force_buffer[:,pi]=((body_ids==bid)[...,None]*applied).sum(1)
  cmd.position_apply_buffer.copy_(robot.data.body_pos_w);robot.has_external_wrench=False
 original_step=cmd.step;callbacks=env.base_env._pre_step_callbacks;callbacks[:]=[external_force if cb==original_step else cb for cb in callbacks];assert external_force in callbacks
 policy=agent.get_rollout_policy('deploy');assert list(policy.in_keys)==['policy']
 video_writer=None
 if a.video_group:
  import omni.usd
  from pxr import UsdGeom,UsdShade,Sdf,Gf
  from isaacsim.core.utils.viewports import set_camera_view
  from PIL import Image,ImageDraw,ImageFont
  import imageio.v2 as imageio
  stage=omni.usd.get_context().get_stage();arrows=[];colors=[(1.,.3,.05),(1.,.75,.05)]
  def material(path,color):
   mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');return mat
  for i,color in enumerate(colors):
   mat=material('/World/TestMaterials/F'+str(i),color);shaft=UsdGeom.Cylinder.Define(stage,f'/World/TestForces/F{i}/shaft');shaft.CreateAxisAttr('Z');shaft.CreateRadiusAttr(.012);head=UsdGeom.Cone.Define(stage,f'/World/TestForces/F{i}/head');head.CreateAxisAttr('Z');head.CreateRadiusAttr(.035)
   for shape in [shaft,head]:UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
   arrows.append((shaft,shaft.AddTransformOp(),head,head.AddTransformOp()))
  def arrow(i,start,force):
   shaft,sop,head,hop=arrows[i];mag=np.linalg.norm(force)
   for shape in [shaft,head]:UsdGeom.Imageable(shape).GetVisibilityAttr().Set('inherited' if mag>.05 else 'invisible')
   if mag<=.05:return
   direction=force/mag;length=.045*mag;tip=min(.08,length*.4);rotation=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction))).GetQuat()
   shaft.CreateHeightAttr(max(.001,length-tip));head.CreateHeightAttr(tip)
   for op,pos in [(sop,start+direction*(length-tip)/2),(hop,start+direction*(length-tip/2))]:
    m=Gf.Matrix4d(1.);m.SetRotate(rotation);m.SetTranslateOnly(Gf.Vec3d(*map(float,pos)));op.Set(m)
  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
  video_writer=imageio.get_writer(a.output/'rollout.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
 def rollout(trials,enabled,trial_index=0):
  global clock_step,force_enabled
  configure(trials);force_enabled=enabled;applied.zero_();clock_step=0;torch.manual_seed(42);np.random.seed(42);td=env.reset();env.base_env.episode_length_buf.zero_()
  initial=hashlib.sha256(robot.data.root_state_w.cpu().numpy().tobytes()+robot.data.joint_pos.cpu().numpy().tobytes()).hexdigest()
  assert (manager.reference.q-robot.data.joint_pos[:,manager.joint_ids]).abs().max()<1e-6
  initial_public=td['policy'].detach().cpu().numpy().copy()
  initial_alpha=manager.low_action_manager.alpha.detach().cpu().numpy().copy()
  initial_delay=manager.low_action_manager.delay.detach().cpu().numpy().copy()
  data={k:[] for k in ['root','point','yaw','height','tilt','force','net_force','yaw_moment','hands_b','joint_pos']};first_fall=torch.full((n,),float('nan'),device=device)
  total_steps=(round((8.+a.sequence_gap+2.)/env.step_dt)+1) if trials[0]['group']=='sequence' else 501
  with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
   for step in range(total_steps):
    if step:
     clock_step=step-1;td=policy(td);td=env.step(td)['next']
    root=robot.data.root_pos_w-env.scene.env_origins;points=robot.data.body_pos_w[torch.arange(n,device=device),body_ids[:,0]]-env.scene.env_origins
    yaw=yaw_from_quat(robot.data.root_quat_w);height=root[:,2];tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
    fall=(height<.45)|(tilt>torch.pi/3);first_fall=torch.where(torch.isnan(first_fall)&fall,torch.full_like(first_fall,step*env.step_dt),first_fall)
    forces=cmd.force_apply_buffer;moment=torch.cross(robot.data.body_pos_w-robot.data.root_pos_w[:,None],forces,dim=-1).sum(1)[:,2]
    assert torch.isfinite(root).all() and torch.isfinite(manager.low_action).all()
    assert applied.norm(dim=-1).max()<=20.001
    if (turns!=0).any():assert forces.sum(1)[turns!=0].abs().max()<1e-5
    assert (forces.norm(dim=-1)>1e-5).sum(-1).max()<=2
    if not enabled:assert forces.count_nonzero()==0
    from active_adaptation.utils.math import quat_apply_inverse
    hands_b=quat_apply_inverse(robot.data.root_quat_w[:,None].expand(-1,2,-1),robot.data.body_pos_w[:,[left_ee,right_ee]]-robot.data.root_pos_w[:,None])
    if hasattr(cmd,'point_force_buffer'):
     torch.testing.assert_close(cmd.point_force_buffer.sum(1),forces.sum(1),atol=1e-5,rtol=1e-5)
    for k,val in dict(root=root,point=points,yaw=yaw,height=height,tilt=tilt,force=applied,net_force=forces.sum(1),yaw_moment=moment,hands_b=hands_b,joint_pos=robot.data.joint_pos).items():data[k].append(val.cpu().numpy().copy())
    if video_writer is not None and step%2==0:
     pos=robot.data.body_pos_w[0,body_ids[0]].cpu().numpy();f=applied[0].cpu().numpy()
     for i in range(2):arrow(i,pos[i],f[i])
     path=np.array(data['root'])[:,0].copy()+env.scene.env_origins[0].cpu().numpy();path[:,2]=.015
     # The measured path is added by decorate_force_suite_video.py.
     center=robot.data.root_pos_w[0].cpu().numpy().copy();center[2]=.72
     actual_root=robot.data.root_pos_w[0].cpu().numpy()
     env.base_env.cfg.viewer.eye=(center+np.array([1.8,-2.2,1.0])-actual_root).tolist()
     env.base_env.cfg.viewer.lookat=(center-actual_root).tolist()
     image=Image.fromarray(env.render('rgb_array'));draw=ImageDraw.Draw(image,'RGBA');draw.rectangle((0,0,1280,145),fill=(10,16,25,230))
     c=trials[0];phase='NO FORCE / INITIAL' if step<100 else 'FORCE PROMPT' if step<300 else 'RELEASED / NO FORCE'
     phase_elapsed=step*env.step_dt
     if c['group']=='sequence':phase='SHOULDER FORCE COUPLE' if 2.0<=phase_elapsed<5.0 else 'EE FORCE COUPLE' if 5.+a.sequence_gap<=phase_elapsed<8.+a.sequence_gap else 'NO FORCE / INITIAL' if phase_elapsed<2 else 'WAIT / NO FORCE' if phase_elapsed<5.+a.sequence_gap else 'RELEASE / NO FORCE'
     draw.text((15,10),f'{"AMP + DENSE POLICY" if cfg.task.command._target_.endswith("AMPComplianceCommand") else "JACOBIAN POLICY"} | {c["name"].upper()} | trial {trial_index+1}/{len(rows)} | {phase_elapsed:.2f}/{(total_steps-1)*env.step_dt:.2f} s',font=font,fill='white')
     force_text=f'shoulders {c["force_N"]:.0f} N/point; EE {a.sequence_ee_force:.1f} N/point' if c['group']=='sequence' else f'{c["force_N"]:.0f} N per point peak'
     draw.text((15,39),f'{phase} | {force_text} | arrows = actual forces (4.5 cm/N)',font=small,fill=(255,210,100))
     delta=data['root'][-1][0]-data['root'][0][0];yaw_delta=np.unwrap(np.array(data['yaw'])[:,0])[-1]-data['yaw'][0][0]
     draw.text((15,65),f'root displacement: X {delta[0]:+.2f} m, Y {delta[1]:+.2f} m | yaw {np.degrees(yaw_delta):+.1f} deg | height {height[0]:.2f} m',font=small,fill='white')
     draw.text((15,91),'Independent standing reset per trial | +X forward, +Y left at start | no motion target | cyan = actual path',font=small,fill=(110,230,255))
     force_label=' | '.join(f'{robot.body_names[int(body_ids[0,i])]}: {np.linalg.norm(f[i]):.1f} N' for i in range(2 if c['turn'] else 1))
     draw.text((15,117),force_label,font=small,fill='white')
     if torch.isfinite(first_fall[0]):draw.text((15,680),f'PHYSICAL FALL at {first_fall[0]:.2f}s; continued without reset',font=small,fill=(255,90,90))
     video_writer.append_data(np.asarray(image))
     if step in [100,250,300,500]:image.save(a.output/f'{c["name"]}_{step:04d}.png')
  data={k:np.stack(v) for k,v in data.items()};data['first_fall_s']=first_fall.cpu().numpy();data['initial_sha256']=initial;data['initial_public']=initial_public;data['initial_alpha']=initial_alpha;data['initial_delay']=initial_delay;data['joint_names']=np.array(robot.joint_names);return data
 if a.video_group:
  summaries=[]
  for i,row in enumerate(rows):
   data=rollout([row],True,i);np.savez_compressed(a.output/(row['name']+'.npz'),**data);summaries.append(dict(row,first_fall_s=float(data['first_fall_s'][0]) if np.isfinite(data['first_fall_s'][0]) else None,initial_sha256=data['initial_sha256']))
   print('VIDEO_TRIAL_DONE',row['name'],flush=True)
  video_writer.close();sequence_steps=round((8.+a.sequence_gap+2.)/env.step_dt)+1;report={'group':a.video_group,'trials':summaries,'seconds':sum(((sequence_steps-1)*env.step_dt if r['group']=='sequence' else 500*env.step_dt) for r in rows),'fps':25,'frames':sum((sequence_steps//2+1 if r['group']=='sequence' else 251) for r in rows),'checkpoint':str(a.checkpoint),'sequence_gap_s':a.sequence_gap if a.video_group=='sequence' else None,'sequence_ee_force_N':a.sequence_ee_force if a.video_group=='sequence' else None}
 elif a.metrics_pass:
  data=rollout(rows,a.metrics_pass=='forced');np.savez_compressed(a.output/(a.metrics_pass+'.npz'),**data)
  report={'pass':a.metrics_pass,'rows':rows,'checkpoint':str(a.checkpoint),'seed':42,'repeats_per_case_and_force':a.repeats,'initial_sha256':data['initial_sha256']}
 else:
  forced=rollout(rows,True);print('FORCED_ROLLOUT_DONE',flush=True);zero=rollout(rows,False);assert forced['initial_sha256']==zero['initial_sha256']
  for key in ['initial_public','initial_alpha','initial_delay']:np.testing.assert_array_equal(forced[key],zero[key])
  preprompt_max_diff=float(np.abs(forced['root'][:101]-zero['root'][:101]).max())
  assert preprompt_max_diff < .005, preprompt_max_diff
  np.savez_compressed(a.output/'forced.npz',**forced);np.savez_compressed(a.output/'zero.npz',**zero)
  records=[]
  for i,row in enumerate(rows):
   direction=np.array(row['vector']);root=forced['root'][:,i];point=forced['point'][:,i];zroot=zero['root'][:,i];zpoint=zero['point'][:,i]
   rd=root[300]-root[100];pd=point[300]-point[100];zr=zroot[300]-zroot[100];zp=zpoint[300]-zpoint[100]
   yaw=np.unwrap(forced['yaw'][:,i]);zyaw=np.unwrap(zero['yaw'][:,i]);yaw_delta=np.degrees(yaw[300]-yaw[100]);zyaw_delta=np.degrees(zyaw[300]-zyaw[100])
   stable=not np.isfinite(forced['first_fall_s'][i]);root_along=float(rd@direction);point_along=float(pd@direction);corrected_root=float((rd-zr)@direction);corrected_point=float((pd-zp)@direction)
   success=stable and ((yaw_delta-zyaw_delta)*row['turn']>10 if row['turn'] else (corrected_root if row['group']=='body' else corrected_point)>.05)
   records.append(dict(row,stable=stable,zero_stable=not np.isfinite(zero['first_fall_s'][i]),first_fall_s=float(forced['first_fall_s'][i]) if not stable else None,root_along_m=root_along,point_along_m=point_along,root_relative_point_along_m=float((pd-rd)@direction),root_along_minus_zero_m=corrected_root,point_along_minus_zero_m=corrected_point,yaw_deg=float(yaw_delta),yaw_minus_zero_deg=float(yaw_delta-zyaw_delta),root_drift_after_release_m=float(np.linalg.norm(root[500,:2]-root[300,:2])),yaw_after_release_deg=float(np.degrees(yaw[500]-yaw[300])),success=bool(success),max_point_force_N=float(np.linalg.norm(forced['force'][:,i],axis=-1).max()),max_net_force_N=float(np.linalg.norm(forced['net_force'][:,i],axis=-1).max())))
  report={'checkpoint':str(a.checkpoint),'seed':42,'repeats_per_case_and_force':a.repeats,'initial_sha256':forced['initial_sha256'],'paired_zero_initial_verified':True,'preprompt_max_root_difference_m':preprompt_max_diff,'records':records,'protocol':'Each trial starts at nominal standing; 0-2s zero, 2-2.5s cosine ramp, 2.5-5.5s hold, 5.5-6s ramp down, 6-10s release. Translation force directions fixed in initial world frame. Turning uses equal/opposite horizontal tangential shoulder forces, following shoulder separation. 150 N/s vector slew. No AMASS guidance/rewards/termination, no within-trial resets. Force-conditioned student, CEER frozen. Success requires no physical fall through 10s and baseline-subtracted movement >5cm (EE for hand trials/root for torso), or signed yaw >10deg. All displacements measured 2-6s; report failures too. Repeats vary stochastic low-level reset parameters, not training seeds.'}
 (a.output/(('pass_'+a.metrics_pass+'.json') if a.metrics_pass else 'summary.json')).write_text(json.dumps(report,indent=2));print('FORCE_SUITE_COMPLETE',str(a.output),flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
