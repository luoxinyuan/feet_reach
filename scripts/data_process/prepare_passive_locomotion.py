"""Select AMASS walking/turning clips and quiet double-foot standing examples.
Kinematic filtering only; no force fitting, contact optimization or inverse dynamics.
"""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active_adaptation.utils.motion import MotionDataset
source=Path('/home/xl521/hiearchical_compliance/dataset/amass')
output=ROOT/'dataset/passive_locomotion_simple'
if output.exists():raise FileExistsError(output)
ds=MotionDataset.create_from_path_lazy(str(source));rng=np.random.default_rng(42)
feet=[ds.body_names.index(s+'_ankle_roll_link') for s in ('left','right')]
buckets={(split,kind):[] for split in ('train','val') for kind in ('move','hold')}
counts={('train','move'):256,('val','move'):64,('train','hold'):64,('val','hold'):16}
rejections={};records=[]
for mid in rng.permutation(ds.num_motions):
 split='val' if mid%5==0 else 'train'
 length=int(ds.lengths[mid]);start=int(ds.starts[mid]);duration=min(300,length-40)
 if duration<180:continue
 first=start+20
 m=ds.data[first:first+duration].clone()
 q=m.root_quat_w.float();q=q/q.norm(dim=-1,keepdim=True)
 tilt=1-2*(q[:,1].square()+q[:,2].square())
 pos=m.root_pos_w.float();vel=m.root_lin_vel_w.float()
 speed=vel[:,:2].norm(dim=-1)
 yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
 yaw_change=float(np.ptp(np.unwrap(yaw.numpy())))
 travel=float((pos[-1,:2]-pos[0,:2]).norm())
 if tilt.min()<.94 or pos[:,2].min()<.55 or pos[:,2].max()>.92 or speed.max()>1.5:continue
 if m.joint_vel.float().abs().max()>12:continue
 if (speed.mean()>.08 and travel>.15) or (yaw_change>.15 and speed.mean()>.025):
  kind='move'
 elif speed.max()<.08 and m.root_ang_vel_w.float().norm(dim=-1).max()<.15 and m.joint_vel.float().abs().max()<.6:
  ankle=m.body_pos_w[:,feet,2].float()
  if (ankle.max(-1).values-ankle.min(-1).values).max()>.025:continue
  kind='hold'
 else:continue
 bucket=buckets[split,kind]
 if len(bucket)>=counts[split,kind]:continue
 # Preserve and normalize the stored G1 geometry. +3.54 cm accounts for sole
 # offset because this legacy dataset places ankle-frame origins on the floor.
 for field in m.__dataclass_fields__:
  value=getattr(m,field)
  if torch.is_floating_point(value):setattr(m,field,value.float())
 offset=m.root_pos_w[0,:2].clone();m.root_pos_w[:,:2]-=offset;m.body_pos_w[:,:,:2]-=offset
 m.root_pos_w[:,2]+=.03540915;m.body_pos_w[:,:,2]+=.03540915
 m.root_quat_w/=m.root_quat_w.norm(dim=-1,keepdim=True)
 m.body_quat_w/=m.body_quat_w.norm(dim=-1,keepdim=True)
 if kind=='hold':
  # Freeze a quiet standing pose, not an arbitrary mid-swing target.
  index=duration//2
  for field in m.__dataclass_fields__:
   value=getattr(m,field);value[:]=value[index].clone()
  m.root_lin_vel_w.zero_();m.root_ang_vel_w.zero_();m.joint_vel.zero_()
 bucket.append(m)
 records.append(dict(source_clip=int(mid),source_start=20,frames=duration,split=split,kind=kind,travel_m=travel,yaw_range_rad=yaw_change))
 if all(len(buckets[k])>=n for k,n in counts.items()):break
for (split,kind),motions in buckets.items():
 if len(motions)<4:raise RuntimeError(f'Insufficient {split}/{kind}: {len(motions)}')
 ends=np.cumsum([len(m) for m in motions]);starts=np.r_[0,ends[:-1]]
 for i,m in enumerate(motions):m.motion_id.fill_(i);m.step[:]=torch.arange(len(m),dtype=m.step.dtype)
 data=torch.cat(motions,dim=0);dest=output/split/kind;dest.mkdir(parents=True)
 data.memmap(str(dest))
 meta=dict(body_names=ds.body_names,joint_names=ds.joint_names,starts=starts.tolist(),ends=ends.tolist(),info={'is_hold':[[float(kind=='hold')]]*len(motions)})
 (dest/'meta_motion.json').write_text(json.dumps(meta))
 print(split,kind,len(motions),int(ends[-1]),flush=True)
report=dict(source=str(source),assumed_fps=50,seed=42,clips=len(records),frames=sum(r['frames'] for r in records),counts={f'{s}/{k}':len(v) for (s,k),v in buckets.items()},records=records)
(output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('PREPARE_OK',report['counts'],flush=True)
