"""Select standing AMASS arm-motion windows for the EE transition experiment.
Only kinematic filtering; preserve retargeted G1 poses without IK or dynamics fitting.
"""
import sys,json,argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active_adaptation.utils.motion import MotionDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dataset/passive_ee_amass')
 p.add_argument('--train-clips',type=int,default=256);p.add_argument('--val-clips',type=int,default=64)
 args=p.parse_args();out=args.output
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2)
 source=Path('/home/xl521/hiearchical_compliance/dataset/amass');ds=MotionDataset.create_from_path_lazy(str(source))
 rng=np.random.default_rng(42);buckets={'train':[],'val':[]};limits={'train':args.train_clips,'val':args.val_clips};records=[];reject=Counter()
 feet=[ds.body_names.index(s+'_ankle_roll_link') for s in ['left','right']]
 hands=[ds.body_names.index(s+'_hand_mimic') for s in ['left','right']]
 lower=[i for i,n in enumerate(ds.joint_names) if any(x in n for x in ['hip','knee','ankle'])]
 thresholds=dict(frames=150,stride=75,root_xy_excursion_m=.06,root_height_range_m=.06,
   root_speed_max_m_s=.20,root_tilt_cos_min=.96,yaw_range_rad=.15,
   foot_excursion_m=.035,foot_vertical_range_m=.025,feet_height_difference_m=.035,
   lower_joint_range_rad=.25,joint_speed_max_rad_s=4.,hand_excursion_min_m=.06)
 for mid in rng.permutation(ds.num_motions):
  split='val' if mid%5==0 else 'train'
  if len(buckets[split])>=limits[split]:continue
  length=int(ds.lengths[mid]);start=int(ds.starts[mid])
  if length<170:continue
  # One accepted window per source clip, to keep long clips from dominating.
  starts=np.arange(10,length-150+1,75);rng.shuffle(starts)
  for offset in starts:
   m=ds.data[start+int(offset):start+int(offset)+150].clone()
   pos=m.root_pos_w.float();q=m.root_quat_w.float();q=q/q.norm(dim=-1,keepdim=True)
   if not torch.isfinite(pos).all() or not torch.isfinite(m.joint_pos).all():reject['nonfinite']+=1;continue
   tilt=1-2*(q[:,1]**2+q[:,2]**2)
   yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
   yaw_range=float(np.ptp(np.unwrap(yaw.numpy())))
   root_excursion=float((pos[:,:2]-pos[0,:2]).norm(dim=-1).max())
   if tilt.min()<.96 or pos[:,2].min()<.60 or pos[:,2].max()>.88 or pos[:,2].max()-pos[:,2].min()>.06 or root_excursion>.06 or yaw_range>.15:
    reject['root_motion']+=1;continue
   foot=m.body_pos_w[:,feet].float();foot_excursion=float((foot-foot[0]).norm(dim=-1).max())
   if foot_excursion>.035 or (foot[:,:,2].max(0).values-foot[:,:,2].min(0).values).max()>.025 or (foot[:,0,2]-foot[:,1,2]).abs().max()>.035:
    reject['feet_motion']+=1;continue
   leg=m.joint_pos[:,lower].float()
   if (leg.max(0).values-leg.min(0).values).max()>.25 or m.joint_vel.float().abs().max()>4. or m.root_lin_vel_w[:,:2].float().norm(dim=-1).max()>.20:
    reject['joint_or_root_velocity']+=1;continue
   hand=m.body_pos_w[:,hands].float();hand_excursion=float((hand-hand[0]).norm(dim=-1).max())
   if hand_excursion<.06:reject['insufficient_arm_motion']+=1;continue
   for field in m.__dataclass_fields__:
    value=getattr(m,field)
    if torch.is_floating_point(value):setattr(m,field,value.float())
   xy=m.root_pos_w[0,:2].clone();m.root_pos_w[:,:2]-=xy;m.body_pos_w[:,:,:2]-=xy
   # Legacy retargeted data places ankle origins on the floor, not soles.
   m.root_pos_w[:,2]+=.03540915;m.body_pos_w[:,:,2]+=.03540915
   m.root_quat_w/=m.root_quat_w.norm(dim=-1,keepdim=True)
   m.body_quat_w/=m.body_quat_w.norm(dim=-1,keepdim=True)
   buckets[split].append(m)
   records.append(dict(split=split,source_clip=int(mid),source_start=int(offset),frames=150,
     root_excursion_m=root_excursion,foot_excursion_m=foot_excursion,hand_excursion_m=hand_excursion,yaw_range_rad=yaw_range))
   if len(records)%20==0:print('SELECTED',len(records),{k:len(v) for k,v in buckets.items()},flush=True)
   break
  if all(len(buckets[k])>=limits[k] for k in buckets):break
 if min(map(len,buckets.values()))<8:raise RuntimeError(f'Too few standing windows: { {k:len(v) for k,v in buckets.items()} }, rejects={reject}')
 for split,motions in buckets.items():
  ends=np.cumsum([len(m) for m in motions]);starts=np.r_[0,ends[:-1]]
  for i,m in enumerate(motions):m.motion_id.fill_(i);m.step[:]=torch.arange(len(m),dtype=m.step.dtype)
  dest=out/split;dest.mkdir(parents=True);torch.cat(motions,dim=0).memmap(str(dest))
  (dest/'meta_motion.json').write_text(json.dumps(dict(body_names=ds.body_names,joint_names=ds.joint_names,starts=starts.tolist(),ends=ends.tolist(),info={})))
 report=dict(source=str(source),assumed_fps=50,seed=42,counts={k:len(v) for k,v in buckets.items()},thresholds=thresholds,
  rejection_counts=dict(reject),records=records,split='source clip ID modulo 5; original AMASS file provenance unavailable',
  normalization='float32; normalize quaternions; initial root xy at zero; add 0.03540915m sole offset')
 (out/'manifest.json').write_text(json.dumps(report,indent=2));print('PREPARE_OK',report['counts'],flush=True)
if __name__=='__main__':main()
