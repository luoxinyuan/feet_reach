#!/usr/bin/env python3
"""Budgeted, non-overlapping behavior-balanced training allowlists; read-only inputs."""
import argparse,bisect,hashlib,json,time
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
import torch
from scan import source,merge
from score_contacts import Geometry

QUOTAS={'single_arm':6.,'both_arms':4.,'locomotion':5.,'yaw':3.,'crouch':3.,'bend':3.,'still':3.,'mixed':3.}
DIRECTIONS=['forward','backward','left','right','up','down']

# Label thresholds are physical units; they do not depend on dataset size.
RULES=dict(samples_per_window=17,hand_range_m=.08,arm_joint_range_rad=.2,
 root_path_m=.25,yaw_range_rad=.30,crouch_height_range_m=.08,crouch_knee_range_rad=.25,
 bend_range_rad=.20,mixed_second_to_first=.65,
 motion_score_scales=dict(hand_m=.25,root_path_m=.5,yaw_rad=.7,height_m=.18,bend_rad=.35),
 source_clip_cap_seconds=60.,pose_hash_joint_resolution_rad=.05,
 notes='Primary labels are heuristics, not human annotations. Direction uses displacement path, not end-minus-start. Arm FK is root-relative. Primary categories have disjoint budgets; auxiliary labels may overlap.')

def heading_vectors(v,yaw):
 """World vectors expressed in heading-only frame."""
 c=np.cos(yaw);s=np.sin(yaw)
 return np.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)

def direction(delta,axes=3):
 energy=np.stack([np.maximum(delta[...,i],0).sum(1) for i in range(axes) for _ in (0,)],-1)
 neg=np.stack([np.maximum(-delta[...,i],0).sum(1) for i in range(axes)],-1)
 weights=np.stack((energy,neg),-1).reshape(len(delta),-1)
 return weights.argmax(-1),weights

def features(q,positions,durations,original_still):
 """Vectorized window features; q [B,17,36], positions wrists + torso basis."""
 B,T,_=q.shape;root=q[...,:3];quat=q[...,3:7];quat=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
 w,x,y,z=np.moveaxis(quat,-1,0);yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=1)
 # Quaternion inverse rotation: hands follow translation/root rotation out of labels.
 off=positions[...,:2,:]-root[:,:,None,:];v=-quat[...,1:][:,:,None,:];qw=quat[...,:1][:,:,None,:]
 t=2*np.cross(v,off);hands=off+qw*t+np.cross(v,t)
 hand_range=np.linalg.norm(np.ptp(hands,axis=1),axis=-1);qj=q[...,7:]
 joint_range=np.ptp(qj,axis=1);arm_range=np.stack((joint_range[:,15:22].max(-1),joint_range[:,22:29].max(-1)),-1)
 active=(hand_range>=.08)&(arm_range>=.2)
 hd=np.diff(hands,axis=1);ldir,lweights=direction(hd[:,:,0]);rdir,rweights=direction(hd[:,:,1])
 root_delta=heading_vectors(np.diff(root,axis=1),yaw[:,:-1]);root_path=np.linalg.norm(root_delta[...,:2],axis=-1).sum(-1)
 rdir_root,rweights_root=direction(root_delta,2);dyaw=np.diff(yaw,axis=1);yaw_range=np.ptp(yaw,axis=1)
 height=np.ptp(root[...,2],axis=1);knee=np.maximum(joint_range[:,3],joint_range[:,9])
 torso=heading_vectors(positions[:,:,3]-positions[:,:,2],yaw)
 lean=np.stack((np.arctan2(torso[...,0],torso[...,2]),np.arctan2(torso[...,1],torso[...,2])),-1)
 lean_range=np.ptp(lean,axis=1);bend=np.max(lean_range,-1);bd=np.diff(lean,axis=1)
 bdir,bweights=direction(np.pad(bd,((0,0),(0,0),(0,1))),2)
 # family order: arms, locomotion, yaw, crouch, bend
 amplitudes=np.stack((np.where(active,hand_range,0).max(-1),root_path,yaw_range,height,bend),-1)
 enabled=np.stack((active.any(-1),root_path>=.25,yaw_range>=.3,(height>=.08)&(knee>=.25),bend>=.2),-1)
 scores=amplitudes/np.array([.25,.5,.7,.18,.35]);scores*=enabled
 order=np.argsort(-scores,axis=1);top=order[:,0];second=order[:,1]
 best=scores[np.arange(B),top];runner=scores[np.arange(B),second]
 mixed=(runner>=.65*best)&(runner>0)
 labels=[];subtypes=[];phase=[];diversity=[];auxiliary=[];fingerprints=[]
 dt=durations[:,None]/(T-1);speed=np.linalg.norm(root_delta[...,:2],axis=-1)/dt
 # Quantized shape signature: pose, relative root path, relative heading; no absolute XY or heading.
 relative=heading_vectors(root-root[:,:1],yaw[:,:1]);relative[...,2]=root[...,2]
 quantized=np.concatenate((np.rint(qj/.05),np.rint(relative/.02),np.rint((yaw-yaw[:,:1])[...,None]/.05)),-1).astype(np.int16)
 for i in range(B):
  aux=[['arms','locomotion','yaw','crouch','bend'][j] for j in range(5) if enabled[i,j]]
  if active[i,0]:aux.append('left_arm')
  if active[i,1]:aux.append('right_arm')
  if active[i].all():aux.append('both_arms')
  if original_still[i]:label='still';aux.append('still')
  elif best[i]<=0:label='unclassified'
  elif mixed[i]:label='mixed'
  else:label=['both_arms' if active[i].all() else 'single_arm','locomotion','yaw','crouch','bend'][top[i]]
  ph='sustained'
  if speed[i,:4].mean()<.05 and speed[i,-4:].mean()>.12:ph='start'
  elif speed[i,-4:].mean()<.05 and speed[i,:4].mean()>.12:ph='stop'
  if label=='single_arm':
   side=0 if active[i,0] else 1;sub=('left' if side==0 else 'right')+'_'+DIRECTIONS[(ldir if side==0 else rdir)[i]]
  elif label=='both_arms':
   den=np.linalg.norm(hd[i,:,0])*np.linalg.norm(hd[i,:,1]);cos=float((hd[i,:,0]*hd[i,:,1]).sum()/max(den,1e-9))
   sub='same_direction' if cos>.5 else 'opposite_direction' if cos<-.3 else 'independent'
  elif label=='locomotion':sub=DIRECTIONS[rdir_root[i]]+'_'+ph
  elif label=='yaw':sub='left' if np.maximum(dyaw[i],0).sum()>=np.maximum(-dyaw[i],0).sum() else 'right'
  elif label=='crouch':sub='rise' if np.maximum(np.diff(root[i,:,2]),0).sum()>=np.maximum(-np.diff(root[i,:,2]),0).sum() else 'lower'
  elif label=='bend':sub=['forward','backward','left','right'][bdir[i]]
  elif label=='still':
   if root[i,:,2].mean()<.55:sub='low_pose'
   elif abs(lean[i]).max()>.3:sub='lean_pose'
   elif hands[i,:,0,2].mean()>.05 or hands[i,:,1,2].mean()>.05:sub='raised_hand_pose'
   else:sub='upright_other'
  elif label=='mixed':sub='+'.join(sorted([['arms','locomotion','yaw','crouch','bend'][top[i]],['arms','locomotion','yaw','crouch','bend'][second[i]]]))
  else:sub='small_motion'
  amp=float(amplitudes[i,top[i]]);strength=float(best[i]);motion_speed=float(np.max(np.mean(np.abs(np.diff(qj[i],axis=0)),axis=0))/(durations[i]/(T-1)))
  amp_bin=int(np.digitize(strength,[.75,1.5,3.]));speed_bin=int(np.digitize(motion_speed,[.1,.4,1.]))
  # Starting posture distinguishes hand heights, root height, knee bend, waist bend.
  pose=[int(np.floor(hands[i,0,0,2]/.2)),int(np.floor(hands[i,0,1,2]/.2)),int(np.floor(root[i,0,2]/.15)),int(np.floor(qj[i,0,3]/.5)),int(np.floor(qj[i,0,14]/.3))]
  labels.append(label);subtypes.append(sub);phase.append(ph);auxiliary.append(aux);diversity.append([amp_bin,speed_bin,*pose])
  fingerprints.append(hashlib.blake2b(quantized[i].tobytes()+str(round(float(durations[i]),4)).encode(),digest_size=12).hexdigest())
 return dict(labels=labels,subtypes=subtypes,phases=phase,auxiliary=auxiliary,diversity=diversity,fingerprints=fingerprints)

@torch.no_grad()
def extract(args):
 geom=Geometry(args.device)
 geom.points=[dict(body=b,offset=x) for b,x in [('left_wrist_yaw_link',[0,0,0]),('right_wrist_yaw_link',[0,0,0]),('torso_link',[0,0,0]),('torso_link',[0,0,1])]]
 records=[];manifests={};start=time.time()
 with (args.output/'candidates.jsonl').open('w') as audit:
  for dataset in ('boneseed','lafan','amass'):
   p=args.input/f'{dataset}.max{args.max_contacts}.allowlist.json';manifest=json.loads(p.read_text());manifests[dataset]=manifest
   for path,stat in manifest['source_identity'].items():assert [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==stat
   allowed={c['clip_id']:c for c in manifest['clips']};iterator,total,fmt=source(Path(manifest['source']),50)
   pending=[];arrays=[]
   def flush():
    if not pending:return
    sampled=np.stack(arrays);batch=torch.as_tensor(sampled.reshape(-1,36),device=args.device)
    positions=geom.fk(batch).cpu().numpy().reshape(len(pending),17,4,3)
    result=features(sampled,positions,np.array([(v['end']-v['start']-1)/50 for v in pending]),np.array(['still' in v['stage1_tags'] for v in pending]))
    for j,row in enumerate(pending):
     row.update(primary=result['labels'][j],subtype=result['subtypes'][j],phase=result['phases'][j],auxiliary=result['auxiliary'][j],diversity=result['diversity'][j],fingerprint=result['fingerprints'][j])
     audit.write(json.dumps(row,separators=(',',':'))+'\n');records.append(row)
    arrays.clear();pending.clear()
   for i,name,q in iterator:
    if i not in allowed:continue
    for w in allowed[i]['windows']:
     t=np.rint(np.linspace(w['start'],w['end']-1,17)).astype(int)
     arrays.append(q[t]);pending.append(dict(dataset=dataset,clip_id=i,source_name=name,start=w['start'],end=w['end'],stage1_tags=w['tags']))
     if len(pending)>=args.batch:flush()
   flush();print('features',dataset,'cumulative windows',len(records),'seconds',round(time.time()-start),flush=True)
 return records,manifests

class IntervalBudget:
 def __init__(self,cap_frames):self.intervals=defaultdict(list);self.used=Counter();self.hashes=set();self.cap=cap_frames
 def accept(self,r,remaining):
  key=(r['dataset'],r['clip_id']);s,e=r['start'],r['end'];n=e-s
  if n>remaining or self.used[key]+n>self.cap or r['fingerprint'] in self.hashes:return False
  spans=self.intervals[key];i=bisect.bisect_left(spans,(s,e))
  if (i>0 and spans[i-1][1]>s) or (i<len(spans) and spans[i][0]<e):return False
  spans.insert(i,(s,e));self.used[key]+=n;self.hashes.add(r['fingerprint']);return True


def select(records,quotas,seed,cap):
 """Scarce categories first; round-robin subtype, source, then feature-bin coverage."""
 rng=np.random.default_rng(seed);order=rng.permutation(len(records));pools={}
 available=Counter();available_sub=Counter()
 for idx in order:
  r=records[int(idx)];cat=r['primary']
  if cat not in quotas:continue
  bucket=pools.setdefault(cat,{}).setdefault(r['subtype'],{}).setdefault(r['dataset'],{})
  bucket.setdefault(tuple(r['diversity']),deque()).append(int(idx));available[cat]+=r['end']-r['start'];available_sub[(cat,r['subtype'])]+=r['end']-r['start']
 budget=IntervalBudget(round(cap*50));chosen=[];counts=Counter();subcounts=Counter();sourcecounts=Counter();bincounts=Counter()
 targets={k:int(v*3600*50) for k,v in quotas.items()}
 # Data scarcity before abundant classes avoids overlap consuming rare categories.
 cats=sorted(quotas,key=lambda c:available[c]/max(targets[c],1))
 for cat in cats:
  pool=pools.get(cat,{})
  while counts[cat]<targets[cat] and pool:
   sub=min(pool,key=lambda s:(subcounts[(cat,s)],s));by_source=pool[sub]
   dataset=min(by_source,key=lambda d:(sourcecounts[(cat,d)],d));bins=by_source[dataset]
   key=min(bins,key=lambda b:(bincounts[(cat,sub,dataset,b)],b));queue=bins[key]
   idx=queue.popleft();r=records[idx];frames=r['end']-r['start']
   if budget.accept(r,targets[cat]-counts[cat]):
    chosen.append(idx);counts[cat]+=frames;subcounts[(cat,sub)]+=frames;sourcecounts[(cat,dataset)]+=frames;bincounts[(cat,sub,dataset,key)]+=frames
   if not queue:del bins[key]
   if not bins:del by_source[dataset]
   if not by_source:del pool[sub]
  print('selected',cat,round(counts[cat]/180000,3),'/',quotas[cat],'h',flush=True)
 return chosen,counts,subcounts,sourcecounts


def write_outputs(args,records,manifests,selected,quotas,elapsed):
 selected_records=[records[i] for i in selected];grouped=defaultdict(list);counts=Counter();sub=Counter();aux=Counter();sources=Counter()
 for r in selected_records:
  grouped[(r['dataset'],r['clip_id'])].append(r);n=r['end']-r['start'];counts[r['primary']]+=n;sub[(r['primary'],r['subtype'])]+=n;sources[r['dataset']]+=n
  for tag in r['auxiliary']:aux[tag]+=n
 total=sum(counts.values());assert total<=int(args.budget_hours*180000)
 files={}
 for dataset,manifest in manifests.items():
  clips=[]
  for (ds,i),rows in sorted(grouped.items()):
   if ds!=dataset:continue
   rows=sorted(rows,key=lambda r:r['start']);intervals=[(r['start'],r['end']) for r in rows]
   assert all(intervals[j][1]<=intervals[j+1][0] for j in range(len(intervals)-1))
   clips.append(dict(clip_id=i,source_name=rows[0]['source_name'],windows=[{k:r[k] for k in ('start','end','primary','subtype','phase','auxiliary','diversity')} for r in rows],intervals=merge(intervals)))
  p=args.output/f'{dataset}.allowlist.json';files[dataset]=str(p.resolve())
  payload={k:manifest[k] for k in ('source','source_format','fps','frame_convention','source_identity','split','split_note')}
  payload.update(schema='compliance-balanced-training-allowlist-v1',stage2_manifest=str((args.input/f'{dataset}.max{args.max_contacts}.allowlist.json').resolve()),role='training candidate only; no independent split certified',summary=dict(clips=len(clips),windows=sum(len(c['windows']) for c in clips),hours=sources[dataset]/180000),clips=clips)
  p.write_text(json.dumps(payload,indent=2)+'\n')
  for path,stat in manifest['source_identity'].items():assert [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==stat
 fingerprints=[r['fingerprint'] for r in selected_records];assert len(fingerprints)==len(set(fingerprints))
 summary=dict(total_hours=total/180000,budget_hours=args.budget_hours,windows=len(selected),quota_hours=quotas,actual_hours={k:counts[k]/180000 for k in quotas},shortfall_hours={k:quotas[k]-counts[k]/180000 for k in quotas},dataset_hours={k:v/180000 for k,v in sources.items()},subtype_hours={f'{k}/{s}':v/180000 for (k,s),v in sub.items()},auxiliary_hours={k:v/180000 for k,v in aux.items()},candidate_primary_counts=dict(Counter(r['primary'] for r in records)),elapsed_seconds=elapsed,validation=dict(no_overlapping_frames=True,no_quantized_duplicate_signatures=True,within_budget=True,source_files_unchanged=True),seed=args.seed,source_clip_cap_seconds=args.clip_cap_seconds)
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Joint manifest is a lightweight index; no motion arrays copied.
 (args.output/'allowlist.json').write_text(json.dumps(dict(schema='compliance-balanced-training-index-v1',datasets=files,summary=summary,rules=RULES),indent=2)+'\n')
 lines=['# Behavior-balanced compliance training allowlist','','| Primary behavior | Target h | Selected h | Shortfall h |','|---|---:|---:|---:|']
 for k in quotas:lines.append(f'| {k} | {quotas[k]:.3f} | {counts[k]/180000:.3f} | {quotas[k]-counts[k]/180000:.3f} |')
 lines+=['',f'Total: {total/180000:.4f} h, {len(selected):,} non-overlapping windows. Source clip contribution <= {args.clip_cap_seconds:g} seconds. Quantized trajectory fingerprints are unique; this is not a guarantee of semantic deduplication.','', '| Dataset | Hours |','|---|---:|']
 for k,v in sources.items():lines.append(f'| {k} | {v/180000:.3f} |')
 lines+=['','## Direction / subtype coverage','','| Behavior / subtype | Hours |','|---|---:|']
 for (c,s),v in sorted(sub.items()):lines.append(f'| {c}/{s} | {v/180000:.3f} |')
 lines+=['','No categories were duplicated or relabelled to fill shortages. Primary categories count disjoint time; auxiliary tags overlap. Source equality is secondary to behavior coverage and available supply. Labels are FK/motion heuristics, not verified compliance response. AMASS original-recording provenance is unavailable; source cap and overlap checks apply to cached clips. Do not use this method-selected training list as a common test set. Intervals index processed 50 Hz sources, not original mocap files. No original data was modified.']
 (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:summary[k] for k in ('total_hours','actual_hours','dataset_hours','elapsed_seconds')},indent=2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('artifacts/compliance_dataset/20261115_stage2_sparse_v1'));p.add_argument('--output',type=Path,required=True);p.add_argument('--max-contacts',type=int,choices=[3,5],default=5);p.add_argument('--budget-hours',type=float,default=30);p.add_argument('--clip-cap-seconds',type=float,default=60);p.add_argument('--seed',type=int,default=20261115);p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=512);p.add_argument('--reuse-candidates',type=Path,help='Read candidates.jsonl from an earlier extraction with matching input/rules')
 a=p.parse_args();assert a.budget_hours>0 and a.clip_cap_seconds>0;a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.time()
 config=dict(rules=RULES,quotas=QUOTAS,input=str(a.input.resolve()),max_contacts=a.max_contacts,seed=a.seed,source_clip_cap_seconds=a.clip_cap_seconds,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 if a.reuse_candidates:
  old=json.loads((a.reuse_candidates.parent/'config.json').read_text());assert old['input']==config['input'] and old['max_contacts']==config['max_contacts'] and old['rules']==config['rules']
  records=[json.loads(line) for line in a.reuse_candidates.open()];manifests={n:json.loads((a.input/f'{n}.max{a.max_contacts}.allowlist.json').read_text()) for n in ('boneseed','lafan','amass')}
 else:records,manifests=extract(a)
 quotas={k:v*a.budget_hours/30 for k,v in QUOTAS.items()};selected,*_=select(records,quotas,a.seed,a.clip_cap_seconds);write_outputs(a,records,manifests,selected,quotas,time.time()-start)
if __name__=='__main__':main()
