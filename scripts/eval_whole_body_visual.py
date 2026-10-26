"""Full-clip diagnostic video: mesh reference ghost, physical forces, no resets."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--controller',default='student',choices=['student','teacher']);p.add_argument('--clip',type=int,default=-1);p.add_argument('--preview',type=int,default=0);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
if args.checkpoint:ckpt=args.checkpoint
else:
 run=Path((ROOT/'outputs/passive-whole-body/latest_pipeline.txt').read_text().strip());phase='finetune' if args.controller=='student' else 'train';ckpt=next((run/phase).rglob('checkpoint_final.pt'))
state=torch.load(ckpt,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
cfg.checkpoint_path=str(ckpt);cfg.vecnorm='eval';cfg.task.num_envs=1;cfg.wandb.mode='disabled';cfg.eval_render=True;cfg.headless=True
cfg.task.command.dataset.mem_paths=['passive_whole_body_full/val'];cfg.task.command.dataset.path_weights=[1.];cfg.task.command.dataset.resample_on_reset=False
if 'guidance_cache' in cfg.task.command:cfg.task.command.guidance_cache=str(ROOT/'artifacts/ceer_segmented_jacobian/val18')
# Choose a longest clip with median root travel, without looking at policy performance.
from active_adaptation.utils.motion import MotionDataset
ds=MotionDataset.create_from_path_lazy(str(ROOT/'dataset/passive_whole_body_full/val'))
long=(ds.lengths==ds.lengths.max()).nonzero().flatten().tolist();travel=[]
for i in long:
 s,e=int(ds.starts[i]),int(ds.ends[i]);x=ds.data.root_pos_w[s:e,:2].float();travel.append(float((x[1:]-x[:-1]).norm(dim=-1).sum()))
clip=args.clip if args.clip>=0 else long[np.argsort(travel)[len(long)//2]]
cfg.task.command.dataset.fix_motion_id=clip;cfg.task.max_episode_length=2000;cfg.task.viewer.resolution=[1280,720]
del ds,state
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=True,
 kit_args='--/plugins/carb.tasking.plugin/threadCount=16 --/plugins/omni.tbb.globalcontrol/maxThreadCount=16').app
try:
 import omni.usd
 from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
 from isaaclab.sim import SimulationContext
 from isaacsim.core.utils.viewports import set_camera_view
 from scripts.utils.helpers import make_env_policy
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 from PIL import Image,ImageDraw,ImageFont
 import imageio.v2 as imageio
 torch.manual_seed(42);np.random.seed(42)
 env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval();cmd=env.command_manager;asset=cmd.asset;cmd.zero_init_prob=1.
 policy=agent.get_rollout_policy('deploy' if args.controller=='student' else 'train')
 if args.controller=='student':assert list(policy.in_keys)==['policy']
 torch.manual_seed(42);np.random.seed(42)
 td=env.reset();assert int(cmd.t[0])==0
 initial_hash=hashlib.sha256(asset.data.joint_pos.cpu().numpy().tobytes()+asset.data.root_pos_w.cpu().numpy().tobytes()).hexdigest()
 label=cfg.task.name if args.checkpoint else args.controller
 stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache()
 def material(path,color,opacity=1.):
  mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
  shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.6);shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(np.array(color)*.12)))
  mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');return mat
 cyan=material('/World/VideoMaterials/Ghost',(.05,.85,1.),.16)
 root=stage.GetPrimAtPath('/World/envs/env_0/Robot');bodyprims={pr.GetName():pr for pr in Usd.PrimRange(root) if pr.GetName() in asset.body_names}
 ghost=[];mesh_count=0
 for name in asset.body_names:
  if name not in bodyprims or name not in cmd.dataset.body_names:continue
  body=bodyprims[name];xf=UsdGeom.Xform.Define(stage,'/World/ReferenceGhost/'+name);op=xf.AddTransformOp();ghost.append((op,cmd.dataset.body_names.index(name)))
  bodyworld=cache.GetLocalToWorldTransform(body)
  for j,prim in enumerate(Usd.PrimRange(body,Usd.TraverseInstanceProxies())):
   if not prim.IsA(UsdGeom.Mesh):continue
   if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':continue
   src=UsdGeom.Mesh(prim);dst=UsdGeom.Mesh.Define(stage,str(xf.GetPath())+'/mesh'+str(j));mesh_count+=1
   for attr in ['points','faceVertexCounts','faceVertexIndices','normals','subdivisionScheme','orientation','doubleSided']:
    a=prim.GetAttribute(attr)
    if a and a.HasAuthoredValueOpinion():dst.GetPrim().CreateAttribute(attr,a.GetTypeName()).Set(a.Get())
   dst.SetNormalsInterpolation(src.GetNormalsInterpolation());dst.CreateDoubleSidedAttr(False)
   UsdGeom.Xformable(dst).AddTransformOp().Set(cache.GetLocalToWorldTransform(prim)*bodyworld.GetInverse())
   UsdShade.MaterialBindingAPI.Apply(dst.GetPrim()).Bind(cyan)
 assert mesh_count>10,mesh_count
 colors=[(1.,.3,.05),(1.,.75,.05),(.95,.15,.65)]
 arrows=[]
 for i,color in enumerate(colors):
  mat=material('/World/VideoMaterials/Force'+str(i),color)
  shaft=UsdGeom.Cylinder.Define(stage,f'/World/Forces/F{i}/shaft');shaft.CreateAxisAttr('Z');shaft.CreateRadiusAttr(.012)
  head=UsdGeom.Cone.Define(stage,f'/World/Forces/F{i}/head');head.CreateAxisAttr('Z');head.CreateRadiusAttr(.035)
  for shape in [shaft,head]:UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
  arrows.append((shaft,shaft.AddTransformOp(),head,head.AddTransformOp()))
 def matrix(pos,quat):
  m=Gf.Matrix4d(1.);m.SetRotate(Gf.Quatd(float(quat[0]),Gf.Vec3d(*map(float,quat[1:]))));m.SetTranslateOnly(Gf.Vec3d(*map(float,pos)));return m
 def arrow(i,start,force):
  shaft,sop,head,hop=arrows[i];mag=float(np.linalg.norm(force));visible=mag>.05
  for shape in [shaft,head]:UsdGeom.Imageable(shape).GetVisibilityAttr().Set('inherited' if visible else 'invisible')
  if not visible:return
  direction=force/mag;length=.08*mag;tip=min(.09,length*.4);q=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction))).GetQuat()
  quat=[q.GetReal(),*q.GetImaginary()];shaft.CreateHeightAttr(max(.001,length-tip));head.CreateHeightAttr(tip)
  sop.Set(matrix(start+direction*(length-tip)/2,quat));hop.Set(matrix(start+direction*(length-tip/2),quat))
 # Reference pelvis path along the floor, covering the entire stored clip.
 full=cmd.dataset.get_slice(None,torch.zeros(1,dtype=torch.long,device=env.device),int(cmd.lengths[0]))
 path=full.root_pos_w[0].cpu().numpy()+env.scene.env_origins[0].cpu().numpy();path[:,2]=.015
 curve=UsdGeom.BasisCurves.Define(stage,'/World/ReferencePath');curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(path)]);curve.CreatePointsAttr([Gf.Vec3f(*map(float,x)) for x in path]);curve.CreateWidthsAttr([.012]);curve.SetWidthsInterpolation('constant');UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material('/World/VideoMaterials/Path',(.05,.7,.85)))
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 length=int(cmd.lengths[0]);steps=length-1 if not args.preview else min(args.preview,length-1);rows=[];first_fail=None;images=0
 writer=imageio.get_writer(args.output/'rollout.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  for step in range(steps+1):
   if step:
    td=policy(td);td=env.step(td)['next']
    if bool(td['terminated'].any()) and first_fail is None:first_fail=step*env.step_dt
   # Never reset after done: render complete reference; mark any early failure.
   pos=cmd._motion.body_pos_w[0,0].cpu().numpy()+env.scene.env_origins[0].cpu().numpy();quat=cmd._motion.body_quat_w[0,0].cpu().numpy()
   if hasattr(cmd,'point_force_buffer'):
    actual=cmd.point_positions_w()[0].cpu().numpy();forces=cmd.point_force_buffer[0].cpu().numpy();force_names=cmd.point_names
   else:
    actual=asset.data.body_pos_w[0].cpu().numpy();forces=cmd.force_apply_buffer[0].cpu().numpy();force_names=asset.body_names
   ids=np.flatnonzero(np.linalg.norm(forces,axis=-1)>.05)
   for op,i in ghost:op.Set(matrix(pos[i],quat[i]))
   for i in range(3):arrow(i,actual[ids[i]] if i<len(ids) else np.zeros(3),forces[ids[i]] if i<len(ids) else np.zeros(3))
   rootpos=asset.data.root_pos_w[0].cpu().numpy();ref=cmd.reward_root_pos_w[0].cpu().numpy();center=(rootpos+ref)/2;center[2]=.75
   distance=max(1.,np.linalg.norm(rootpos-ref)/2)
   env.base_env.cfg.viewer.eye=(center+np.array([1.7,-2.,.95])*distance-rootpos).tolist()
   env.base_env.cfg.viewer.lookat=(center-rootpos).tolist()
   error=float((asset.data.body_pos_w[:,cmd.keypoint_idx_asset]-cmd.reward_keypoints_w[:,cmd.keypoint_idx_motion]).norm(dim=-1).mean())
   rows.append(dict(time_s=step*env.step_dt,reference_frame=int(cmd.t[0]),keypoint_error_m=error,force=forces.tolist(),points=actual.tolist()))
   if step%2==0 or step==steps:
    rgb=env.render('rgb_array');im=Image.fromarray(rgb);draw=ImageDraw.Draw(im,'RGBA')
    # Project the same 3D vectors as an on-top overlay so tiny forces remain legible.
    camera=UsdGeom.Camera(stage.GetPrimAtPath('/OmniverseKit_Persp')).GetCamera(Usd.TimeCode.Default())
    vp=camera.frustum.ComputeViewMatrix()*camera.frustum.ComputeProjectionMatrix()
    def project(x):
     v=Gf.Vec4d(*map(float,x),1.)*vp
     return np.array([(float(v[0]/v[3])*.5+.5)*1280,(.5-float(v[1]/v[3])*.5)*720])
    for i,idx in enumerate(ids[:3]):
     a=project(actual[idx]);b=project(actual[idx]+forces[idx]*.08);delta=b-a;norm=np.linalg.norm(delta);color=tuple(int(c*255) for c in colors[i])
     if norm>2:
      u=delta/norm;n=np.array([-u[1],u[0]]);draw.line([tuple(a),tuple(b)],fill=color,width=5);draw.polygon([tuple(b),tuple(b-u*13+n*7),tuple(b-u*13-n*7)],fill=color);draw.ellipse((a[0]-5,a[1]-5,a[0]+5,a[1]+5),fill=color)
    draw.rectangle((0,0,1280,112),fill=(12,18,28,225))
    draw.text((16,10),f'{label.upper()} | clip {clip} | frame {int(cmd.t[0])}/{length-1} | {step*env.step_dt:.2f} / {(length-1)*env.step_dt:.2f} s',font=font,fill='white')
    draw.text((16,39),'CYAN TRANSLUCENT ROBOT = reference pose | SOLID ROBOT = actual | CYAN FLOOR LINE = reference root path',font=small,fill=(100,230,255))
    draw.text((16,64),f'ARROWS = applied external forces (8 cm/N) | total {np.linalg.norm(forces,axis=-1).sum():.2f} N | keypoint error {error*100:.1f} cm',font=small,fill='white')
    draw.text((16,87),'FULL CLIP FROM FRAME 0 | real-time 1x | no resets or cuts',font=small,fill='white')
    draw.rectangle((12,125,390,235),fill=(12,18,28,210))
    for i,idx in enumerate(ids[:3]):draw.text((22,133+29*i),f'{force_names[idx]}: {np.linalg.norm(forces[idx]):.2f} N',font=small,fill=tuple(int(c*255) for c in colors[i]))
    if first_fail is not None:draw.text((16,680),f'TRACKING TERMINATION at {first_fail:.2f}s; continuing WITHOUT RESET for diagnosis',font=small,fill=(255,100,90))
    writer.append_data(np.asarray(im));images+=1
    if step in (0,50,250,500,750):im.save(args.output/f'frame_{step:04d}.png')
 writer.close()
 summary=dict(controller=args.controller,backend=label,seed=42,initial_sha256=initial_hash,clip_id=clip,selection='longest clip, median root travel; no performance filtering' if args.clip<0 else 'explicit clip',checkpoint=str(ckpt),reference_frames=length,rendered_reference_end=int(cmd.t[0]),video_frames=images,video_fps=25,video_seconds=images/25,first_tracking_failure_s=first_fail,ghost_meshes=mesh_count,ghost_opacity=.16,force_arrow_m_per_N=.08,reset_count=1)
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2));(args.output/'trace.json').write_text(json.dumps(rows));print('VIDEO_COMPLETE',summary,flush=True)
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();os._exit(1)
else:app.close()
