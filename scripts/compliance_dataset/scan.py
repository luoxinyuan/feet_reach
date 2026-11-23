#!/usr/bin/env python3
"""Read-only common G1 quality scan. Outputs window allowlists, never motion data.

Frame spans are clip-local [start,end) in the specified 50-Hz processed source,
NOT original CSV/AMASS frame numbers or generate_dataset.py segment filters.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

DEFAULTS = {
 'boneseed': '/home/xl521/supermimic_force_transition/data/processed/bones_seed_raw_50hz.npz',
 'lafan': '/home/xl521/supermimic_force_transition/data/processed/lafan1_50hz.npz',
 'amass': '/home/xl521/hiearchical_compliance/dataset/amass',
}
# Conservative corruption/scope gates; not certified hardware joint limits.
THRESHOLDS = dict(quat_norm_error=.05, joint_abs_rad=6.4,
 joint_step_rad=.6, root_step_m=.25, root_rotation_step_rad=.7,
 root_height_min_m=.15, root_height_max_m=1.5, root_tilt_max_deg=100.,
 fast_joint_speed_rad_s=8., fast_root_speed_m_s=2., fast_acceleration_rad_s2=150.)

def windows(n, fps, seconds=4., stride=2., minimum=1.):
 w=max(2,round(seconds*fps)); step=max(1,round(stride*fps))
 if n < max(3,round(minimum*fps)): return []
 if n<=w:return [(0,n)]
 starts=list(range(0,n-w+1,step))
 if starts[-1]!=n-w:starts.append(n-w)
 return [(s,s+w) for s in starts]

def merge(spans):
 out=[]
 for s,e in sorted(spans):
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out

def inspect(qpos,fps,thresholds=THRESHOLDS):
 t=thresholds; reasons=[]; tags=[]
 if not np.isfinite(qpos).all():return ['nonfinite'],[],{}
 pos=qpos[:,:3].astype(np.float64); quat=qpos[:,3:7].astype(np.float64); q=qpos[:,7:].astype(np.float64)
 norm=np.linalg.norm(quat,axis=1)
 if np.max(abs(norm-1))>t['quat_norm_error']:reasons.append('quaternion_norm')
 quat=quat/np.maximum(norm[:,None],1e-12)
 dq=np.diff(q,axis=0); dp=np.diff(pos,axis=0)
 rotation=2*np.arccos(np.clip(abs(np.sum(quat[1:]*quat[:-1],axis=1)),0,1))
 tilt=np.degrees(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
 # 100ms differencing reduces float16 quantization artifacts in AMASS.
 h=max(1,round(.1*fps)); vel=(q[h:]-q[:-h])/(h/fps)
 accel=np.diff(vel,axis=0)*fps
 metrics={'joint_step_max_rad':float(abs(dq).max()),'root_step_max_m':float(np.linalg.norm(dp,axis=1).max()),
 'root_rotation_step_max_rad':float(rotation.max()),'root_height_min_m':float(pos[:,2].min()),
 'root_height_max_m':float(pos[:,2].max()),'root_tilt_max_deg':float(tilt.max()),
 'joint_speed_p95_rad_s':float(np.percentile(np.max(abs(vel),axis=1),95)),
 'joint_acceleration_p95_rad_s2':float(np.percentile(np.max(abs(accel),axis=1),95)),
 'root_speed_p95_m_s':float(np.percentile(np.linalg.norm((pos[h:]-pos[:-h])/(h/fps),axis=1),95)),
 'root_travel_m':float(np.linalg.norm(pos[-1,:2]-pos[0,:2])), 'height_range_m':float(np.ptp(pos[:,2]))}
 checks=[('joint_abs',abs(q).max()>t['joint_abs_rad']),('joint_jump',metrics['joint_step_max_rad']>t['joint_step_rad']),
 ('root_jump',metrics['root_step_max_m']>t['root_step_m']),('rotation_jump',rotation.max()>t['root_rotation_step_rad']),
 ('root_height',pos[:,2].min()<t['root_height_min_m'] or pos[:,2].max()>t['root_height_max_m']),
 ('root_tilt',tilt.max()>t['root_tilt_max_deg'])]
 reasons += [name for name,bad in checks if bad]
 # Standard G1 MuJoCo ordering: legs12, waist3, left arm7, right arm7.
 left=float(np.max(np.ptp(q[:,15:22],axis=0)));right=float(np.max(np.ptp(q[:,22:29],axis=0)))
 metrics.update(left_arm_joint_range_rad=left,right_arm_joint_range_rad=right)
 yaw=np.unwrap(np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2)))
 metrics['yaw_range_rad']=float(np.ptp(yaw))
 if left>.25:tags.append('left_arm')
 if right>.25:tags.append('right_arm')
 if left>.25 and right>.25:tags.append('both_arms')
 if metrics['root_travel_m']>.25:tags.append('locomotion')
 if np.ptp(yaw)>.3:tags.append('yaw')
 if metrics['height_range_m']>.10:tags.append('vertical_motion')
 # Heuristic crouch: height excursion together with knee flexion excursion.
 if metrics['height_range_m']>.10 and max(np.ptp(q[:,3]),np.ptp(q[:,9]))>.3:tags.append('crouch')
 if max(np.ptp(q[:,13]),np.ptp(q[:,14]))>.2 or np.ptp(tilt)>15:tags.append('bend')
 if np.max(np.ptp(q,axis=0))<.10 and np.max(np.ptp(pos,axis=0))<.03 and np.ptp(yaw)<.05:tags.append('still')
 if metrics['joint_speed_p95_rad_s']>t['fast_joint_speed_rad_s'] or metrics['root_speed_p95_m_s']>t['fast_root_speed_m_s'] or metrics['joint_acceleration_p95_rad_s2']>t['fast_acceleration_rad_s2']:tags.append('fast')
 if not tags:tags=['small_motion']
 return reasons,tags,metrics

def source(path,fps):
 if path.is_file():
  d=np.load(path,allow_pickle=False);actual=float(d['fps'])
  if abs(actual-fps)>1e-6:raise ValueError(f'FPS mismatch {actual} != {fps}')
  print(f'Loading qpos only: {path}',flush=True);q=d['qpos'];starts=d['starts'];lengths=d['lengths'];names=d['names'];d.close()
  assert q.shape[1]==36
  def iterator():
   for i,(s,n,name) in enumerate(zip(starts,lengths,names)):
    yield i,str(name),q[int(s):int(s+n)]
  return iterator(),len(starts),'npz_qpos_wxyz_g1_29'
 meta=json.loads((path/'meta_motion.json').read_text());spec=json.loads((path/'_tensordict/meta.json').read_text())
 assert len(meta['joint_names'])==29
 expected={3:'left_knee_joint',9:'right_knee_joint',15:'left_shoulder_pitch_joint',22:'right_shoulder_pitch_joint'}
 for i,name in expected.items():assert meta['joint_names'][i]==name
 arrays={}
 for k in ['root_pos_w','root_quat_w','joint_pos']:
  s=spec[k];arrays[k]=np.memmap(path/'_tensordict'/f'{k}.memmap',mode='r',dtype=s['dtype'].removeprefix('torch.'),shape=tuple(s['shape']))
 def iterator():
  for i,(s,e) in enumerate(zip(meta['starts'],meta['ends'])):
   yield i,f'cache_motion_{i}',np.concatenate([arrays[k][s:e] for k in ['root_pos_w','root_quat_w','joint_pos']],axis=1).astype(np.float32)
 return iterator(),len(meta['starts']),'motion_memmap_wxyz_g1_29'

def run(name,path,args):
 before={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in ([path] if path.is_file() else [path/'meta_motion.json',path/'_tensordict/meta.json',*sorted((path/'_tensordict').glob('*.memmap'))])}
 it,total,fmt=source(path,args.fps);clips=[];rejected=Counter();labels=Counter();count=accepted=frames=kept_frames=short=0
 with (args.output/f'{name}.audit.jsonl').open('w') as audit:
  for i,source_name,q in it:
   frames+=len(q);spans=[];accepted_windows=[]
   ws=windows(len(q),args.fps,args.window,args.stride,args.min_duration)
   if not ws:short+=1
   for s,e in ws:
    reasons,tags,metrics=inspect(q[s:e],args.fps,args.thresholds);count+=1
    audit.write(json.dumps(dict(clip_id=i,start=s,end=e,accepted=not reasons,reasons=reasons,tags=tags,metrics=metrics),allow_nan=False)+'\n')
    if reasons:rejected.update(reasons)
    else:
     accepted+=1;labels.update(tags);spans.append((s,e));accepted_windows.append(dict(start=s,end=e,tags=tags))
   if spans:
    union=merge(spans);kept_frames+=sum(e-s for s,e in union)
    clips.append(dict(clip_id=i,source_name=source_name,windows=accepted_windows,intervals=union))
   if (i+1)%10000==0:print(f'{name}: {i+1}/{total}, accepted {accepted}/{count}',flush=True)
 after={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in before}
 assert before==after,'Source changed during scan'
 summary=dict(dataset=name,source_clips=total,allowed_clips=len(clips),short_clips=short,windows=count,allowed_windows=accepted,
              source_hours=frames/args.fps/3600,allowed_union_hours=kept_frames/args.fps/3600,rejected_reasons=dict(rejected),accepted_tags=dict(labels),source_unchanged=True)
 payload=dict(schema='compliance-window-allowlist-v1',source=str(path.resolve()),source_format=fmt,fps=args.fps,
 frame_convention='clip-local [start,end), processed source frames; intervals are unions, windows retain scan boundaries',
 params=dict(window_seconds=args.window,stride_seconds=args.stride,min_duration_seconds=args.min_duration,thresholds=args.thresholds),
 source_identity=before,split=None,split_note='No split assigned: AMASS cache has no original recording provenance; group by original recording before train/test split.',
 summary=summary,clips=clips)
 (args.output/f'{name}.allowlist.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary),flush=True);return summary

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name,path in DEFAULTS.items():p.add_argument('--'+name,type=Path,default=Path(path))
 p.add_argument('--datasets',nargs='+',choices=list(DEFAULTS),default=list(DEFAULTS));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--fps',type=float,default=50);p.add_argument('--window',type=float,default=4);p.add_argument('--stride',type=float,default=2);p.add_argument('--min-duration',type=float,default=1)
 p.add_argument('--thresholds',type=Path,help='JSON overrides of documented thresholds')
 a=p.parse_args();assert a.fps>0 and a.window>=a.min_duration>=.2 and a.stride>0
 override=json.loads(a.thresholds.read_text()) if a.thresholds else {};assert not set(override)-set(THRESHOLDS)
 a.thresholds={**THRESHOLDS,**override};a.output.mkdir(parents=True,exist_ok=True)
 if any((a.output/f'{n}.allowlist.json').exists() for n in a.datasets):raise FileExistsError('Use a fresh output directory')
 start=time.time();summaries=[run(n,getattr(a,n),a) for n in a.datasets]
 (a.output/'summary.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-start,datasets=summaries,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
 lines=['# Common G1 compliance quality allowlists','', '| Dataset | Source clips | Allowed clips | Allowed windows / total | Union hours / source hours |','|---|---:|---:|---:|---:|']
 for s in summaries:lines.append(f"| {s['dataset']} | {s['source_clips']} | {s['allowed_clips']} | {s['allowed_windows']} / {s['windows']} | {s['allowed_union_hours']:.2f} / {s['source_hours']:.2f} |")
 lines+=['','Only conservative quality/scope gates. Fast motions are tagged, not rejected. No dynamics, contact/collision, joint-limit certification, low-level tracking or force-expression tests. Arm tags use joint excursions, not Cartesian EE motion. No split assigned because AMASS recording provenance is absent. No original dataset modified. Existing generate_dataset.py exact-segment allowlists are a DIFFERENT schema; consume this manifest using clip_id and local spans.']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
