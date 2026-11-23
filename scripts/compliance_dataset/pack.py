#!/usr/bin/env python3
"""Pack ComplianceMix selected windows into native MotionData memmaps (no force cache)."""
import argparse,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active_adaptation.utils.motion import MotionData,G1_JOINT_NAMES
from active_adaptation.utils.surface_fk import SurfaceFK
from active_adaptation.utils.surface_guidance import quat_apply
from scan import source
from balance import QUOTAS

FPS=50

def derivatives(qpos):
 """No differencing across clip boundaries, no zero velocity padding."""
 p=qpos[:,:3].astype(np.float64);q=qpos[:,3:7].astype(np.float64);j=qpos[:,7:].astype(np.float64)
 r=Rotation.from_quat(q[:,[1,2,3,0]])
 linear=np.gradient(p,1/FPS,axis=0,edge_order=1);joint=np.gradient(j,1/FPS,axis=0,edge_order=1)
 angular=np.zeros((len(qpos),3),np.float64)
 angular[1:-1]=(r[2:]*r[:-2].inv()).as_rotvec()*(FPS/2)
 angular[0]=(r[1]*r[0].inv()).as_rotvec()*FPS;angular[-1]=(r[-1]*r[-2].inv()).as_rotvec()*FPS
 return linear.astype(np.float32),angular.astype(np.float32),joint.astype(np.float32)

def normalized_qpos(qpos):
 q=np.array(qpos,dtype=np.float32,copy=True);quat=q[:,3:7];quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
 # Quaternion signs are representational only; fix discontinuities.
 sign=np.ones(len(q),np.float32);sign[1:]=np.cumprod(np.where((quat[1:]*quat[:-1]).sum(-1)<0,-1.,1.))
 quat*=sign[:,None];q[:,:2]-=q[0,:2].copy();return q

def allocate(path,total,bodies):
 fields={'motion_id':(torch.int32,()),'step':(torch.int32,()),'root_pos_w':(torch.float32,(3,)),
 'root_quat_w':(torch.float32,(4,)),'root_lin_vel_w':(torch.float32,(3,)), 'root_ang_vel_w':(torch.float32,(3,)),
 'joint_pos':(torch.float32,(29,)),'joint_vel':(torch.float32,(29,)),
 'body_pos_w':(torch.float32,(bodies,3)),'body_pos_b':(torch.float32,(bodies,3)),'body_quat_w':(torch.float32,(bodies,4))}
 template=MotionData(**{k:torch.empty((total,*shape),device='meta',dtype=dtype) for k,(dtype,shape) in fields.items()},batch_size=[total])
 return template.memmap_like(str(path))

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allowlist',type=Path,default=ROOT/'artifacts/compliance_dataset/compliance_mix_29h_v1/allowlist.json');p.add_argument('--output',type=Path,default=ROOT/'dataset/compliance_mix_29h_v1');p.add_argument('--device',default='cuda:0');p.add_argument('--chunk-frames',type=int,default=8192)
 a=p.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False);started=time.time()
 index=json.loads(a.allowlist.read_text());manifests={ds:json.loads(Path(path).read_text()) for ds,path in index['datasets'].items()}
 geometry_path=ROOT/'artifacts/ceer_segmented_jacobian/geometry.json';geometry=json.loads(geometry_path.read_text());fk=SurfaceFK(geometry,G1_JOINT_NAMES,a.device)
 ankles=[fk.bodies.index(x) for x in ('left_ankle_roll_link','right_ankle_roll_link')]
 plans=defaultdict(list);by_source=defaultdict(list);offsets=Counter();sources=list(manifests)
 for ds,m in manifests.items():
  for path,stat in m['source_identity'].items():assert [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==stat
  for c in m['clips']:
   for w in c['windows']:
    category=w['primary'];n=w['end']-w['start'];assert n>=2
    row=dict(dataset=ds,source_name=c['source_name'],source_clip_id=c['clip_id'],source_start=w['start'],source_end=w['end'],category=category,subtype=w['subtype'],auxiliary=w['auxiliary'],motion_id=len(plans[category]),start=offsets[category],end=offsets[category]+n)
    offsets[category]+=n;plans[category].append(row);by_source[(ds,c['clip_id'])].append(row)
 assert sum(offsets.values())==round(index['summary']['total_hours']*180000)
 data={cat:allocate(a.output/'train'/cat,n,len(fk.bodies)) for cat,n in offsets.items()}
 pending=[];qparts=[];n_pending=0;written=0;stats=dict(root_shift_z_min=float('inf'),root_shift_z_max=-float('inf'),max_joint_velocity=0.,max_root_speed=0.)
 def flush():
  nonlocal n_pending,written
  if not pending:return
  batch=torch.tensor(np.concatenate(qparts),device=a.device);P,R=fk(batch[:,7:],batch[:,:3],batch[:,3:7]);P=P.cpu();R=R.cpu();batch=batch.cpu();cursor=0
  for row,original in zip(pending,qparts):
   n=len(original);sl=slice(cursor,cursor+n);q=batch[sl].clone();pos=P[sl].clone();rot=R[sl]
   # Match existing generate_dataset floor convention: min ankle origin z=0;
   # standard loader adds +0.035m to root and all bodies at runtime.
   dz=-float(pos[:,ankles,2].min());q[:,2]+=dz;pos[:,:,2]+=dz
   inv=q[:,3:7].clone();inv[:,1:]*=-1;body_b=quat_apply(inv[:,None,:].expand(-1,len(fk.bodies),-1),pos-q[:,:3,None].transpose(1,2))
   linear,angular,joint=derivatives(q.numpy());dest=data[row['category']];out=slice(row['start'],row['end'])
   fields=dict(root_pos_w=q[:,:3],root_quat_w=q[:,3:7],root_lin_vel_w=torch.from_numpy(linear),root_ang_vel_w=torch.from_numpy(angular),joint_pos=q[:,7:],joint_vel=torch.from_numpy(joint),body_pos_w=pos,body_pos_b=body_b,body_quat_w=rot)
   for key,value in fields.items():
    if not torch.isfinite(value).all():raise ValueError(f'Nonfinite {row} {key}')
    getattr(dest,key)[out]=value
   dest.motion_id[out]=row['motion_id'];dest.step[out]=torch.arange(n,dtype=torch.int32)
   row['translation_applied_xyz']=[row.pop('xy_shift')[0],row.pop('xy_shift_y'),dz]
   stats['root_shift_z_min']=min(stats['root_shift_z_min'],dz);stats['root_shift_z_max']=max(stats['root_shift_z_max'],dz)
   stats['max_joint_velocity']=max(stats['max_joint_velocity'],float(np.abs(joint).max()));stats['max_root_speed']=max(stats['max_root_speed'],float(np.linalg.norm(linear,axis=-1).max()))
   written+=n;cursor+=n
  pending.clear();qparts.clear();n_pending=0
 for ds,m in manifests.items():
  iterator,total,fmt=source(Path(m['source']),50)
  for clip_id,name,qraw in iterator:
   for row in by_source.get((ds,clip_id),[]):
    q=qraw[row['source_start']:row['source_end']];row['xy_shift']=[-float(q[0,0]),-float(q[0,1])];row['xy_shift_y']=-float(q[0,1]);q=normalized_qpos(q)
    pending.append(row);qparts.append(q);n_pending+=len(q)
    if n_pending>=a.chunk_frames:flush()
  flush();print(ds,'written frames',written,'seconds',round(time.time()-started),flush=True)
 assert written==sum(offsets.values())
 report={};weights=[];mem_paths=[]
 for cat in QUOTAS:
  if cat not in plans:continue
  rows=plans[cat];path=a.output/'train'/cat
  meta=dict(body_names=fk.bodies,joint_names=G1_JOINT_NAMES,starts=[r['start'] for r in rows],ends=[r['end'] for r in rows],fps=50,info=dict(primary_id=[[list(QUOTAS).index(cat)] for r in rows],source_dataset_id=[[sources.index(r['dataset'])] for r in rows]))
  (path/'meta_motion.json').write_text(json.dumps(meta)+'\n');(path/'provenance.json').write_text(json.dumps(rows,indent=2)+'\n')
  report[cat]=dict(frames=offsets[cat],clips=len(rows),hours=offsets[cat]/180000,min_frames=min(r['end']-r['start'] for r in rows),max_frames=max(r['end']-r['start'] for r in rows))
  weights.append(offsets[cat]/sum(offsets.values()));mem_paths.append(str(path.relative_to(ROOT/'dataset')) if path.is_relative_to(ROOT/'dataset') else str(path.resolve()))
 config=dict(mem_paths=mem_paths,path_weights=weights,sample_once=True,resample_on_reset=True,position_z_offset=.035,clamp_joint_targets=False,buffer_float_dtype='float16')
 (a.output/'dataset_config.json').write_text(json.dumps(config,indent=2)+'\n')
 summary=dict(name='ComplianceMix-29h-v1',frames=written,hours=written/180000,clips=sum(v['clips'] for v in report.values()),categories=report,body_count=len(fk.bodies),joint_count=29,storage_dtype='float32',fps=50,seconds=time.time()-started,source_allowlist=str(a.allowlist.resolve()),source_allowlist_sha256=hashlib.sha256(a.allowlist.read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stats=stats,conventions='World Z-up, meters/radians, wxyz. Root XY recentered per window; min ankle-roll origin z set to zero, loader adds .035 m. Full USD FK 33 bodies; normalized quaternion signs; within-window central differences and one-sided endpoint velocities. No padding/no joint clamping/no force cache. No train/val split invented.')
 for ds,m in manifests.items():
  for path,stat in m['source_identity'].items():assert [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==stat
 (a.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
 (a.output/'READY').write_text('Packing finished. See validation.json for loader/FK/physics checks.\n')
if __name__=='__main__':main()
