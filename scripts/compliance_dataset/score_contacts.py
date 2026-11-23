#!/usr/bin/env python3
"""Stage 2: sparse upper-body geometric observability proxy, not a decoder eval.
Read-only sources; consumes stage-1 windows and emits separate max3/max5 lists.
"""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scan import source, merge
ROOT=Path(__file__).resolve().parents[2]

def mul(a,b):
 return torch.cat((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True),a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)),-1)
def rotate(q,v):
 t=2*torch.cross(q[...,1:],v,dim=-1);return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)
def rotvec(a,b):
 ac=a.clone();ac[...,1:]*=-1;r=mul(b,ac);r=torch.where(r[...,:1]<0,-r,r);s=r[...,1:].norm(dim=-1,keepdim=True)
 return r[...,1:]*2*torch.atan2(s,r[...,:1].clamp_min(0))/s.clamp_min(1e-9)

class Geometry:
 def __init__(self,device):
  self.device=device;self.geometry=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
  self.names=json.loads((ROOT/'artifacts/force_mapping_eval/20261111_v2_root_valid/manifest.json').read_text())['joint_names']
  assert len(self.names)==29
  # Fixed reproducible surface candidate atlas, independent of dataset and motion.
  mesh=ROOT/'artifacts/surface_compliance/geometry';manifest=json.loads((mesh/'manifest.json').read_text());tri=np.load(mesh/'triangles.npy',mmap_mode='r')
  self.points=[];rng=np.random.default_rng(20261115)
  for part in manifest['parts']:
   if any(k in part['body'] for k in ('hip','knee','ankle')):continue
   triangles=tri[part['start']:part['end']];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)
   for index in rng.choice(len(triangles),2,replace=False,p=area/area.sum()):
    self.points.append(dict(body=part['body'],offset=triangles[index].mean(0).tolist()))
  self.edges=[]
  for e in self.geometry['edges']:
   def tensor(x):return torch.tensor(x,device=device,dtype=torch.float32)
   self.edges.append((e,tensor(e['p0']),tensor(e['p1']),tensor(e['r0']),tensor(e['r1'])))
  self.cols=torch.tensor(list(range(6))+list(range(18,35)),device=device)
  self.scale=torch.tensor([.1]*3+[.3]*3+[.3]*17,device=device)
 def fk(self,qpos,jacobian=False):
  rp=qpos[:,:3];rq=qpos[:,3:7];rq=rq/rq.norm(dim=-1,keepdim=True).clamp_min(1e-9);q=qpos[:,7:];n=len(q)
  states={'pelvis':(rp,rq,[])};joints={};pending=list(self.edges)
  while pending:
   ready=[r for r in pending if r[0]['parent'] in states]
   if not ready:raise ValueError('Disconnected geometry')
   for record in ready:
    e,p0,p1,r0,r1=record;p,r,anc=states[e['parent']];jrot=mul(r,r0.expand(n,4));jp=p+rotate(r,p0.expand(n,3))
    if e['axis']:
     idx=self.names.index(e['name']);axis=q.new_zeros(3);axis['XYZ'.index(e['axis'])]=1;joints[idx]=(jp,rotate(jrot,axis.expand(n,3)));anc=anc+[idx]
     angle=q[:,idx:idx+1]/2;turn=torch.cat((angle.cos(),angle.sin()*axis),-1)
    else:turn=q.new_zeros(n,4);turn[:,0]=1
    inv=r1.clone();inv[1:]*=-1;cr=mul(mul(jrot,turn),inv.expand(n,4));cp=jp-rotate(cr,p1.expand(n,3));states[e['child']]=(cp,cr,anc);pending.remove(record)
  positions=[];jacs=[];eye=torch.eye(3,device=q.device).expand(n,3,3)
  for point in self.points:
   p,r,anc=states[point['body']];local=q.new_tensor(point['offset']).expand(n,3);p=p+rotate(r,local);positions.append(p)
   if jacobian:
    J=q.new_zeros(n,3,35);J[:,:,:3]=eye;J[:,:,3:6]=torch.cross(eye,(p-rp)[:,None,:].expand(n,3,3),dim=-1).transpose(1,2)
    for idx in anc:
     jp,axis=joints[idx];J[:,:,6+idx]=torch.cross(axis,p-jp,dim=-1)
    jacs.append(J[:,:,self.cols]*self.scale)
  P=torch.stack(positions,1)
  return (P,torch.stack(jacs,1)) if jacobian else P

@torch.no_grad()
def score(geom,q0,qh,q1,q2,thresholds):
 P,J=geom.fk(q0,True);P1=geom.fk(q1);P2=geom.fk(q2)
 delta=torch.cat((qh[:,:3]-q0[:,:3],rotvec(q0[:,3:7],qh[:,3:7]),qh[:,7:]-q0[:,7:]),-1)[:,geom.cols]/geom.scale
 n,p,_,d=J.shape;rows=torch.arange(n,device=J.device);I=torch.eye(d,device=J.device).expand(n,d,d);projection=torch.zeros_like(I);chosen=[];result={}
 valid=P[:,:,2]>.06
 for k in range(1,6):
  B=J@(I-projection)[:,None];v=(B@delta[:,None,:,None]);H=B@B.transpose(-1,-2)+1e-5*torch.eye(3,device=J.device)
  gain=(v.transpose(-1,-2)@torch.linalg.solve(H,v)).flatten(2).squeeze(-1)
  gain.masked_fill_(~valid,-torch.inf)
  if chosen:gain.scatter_(1,torch.stack(chosen,1),-torch.inf)
  index=gain.argmax(1);chosen.append(index);b=B[rows,index];projection+=b.transpose(1,2)@torch.linalg.solve(H[rows,index],b)
  ids=torch.stack(chosen,1);A=J[rows[:,None],ids].flatten(1,2);y=(A@delta[...,None]).squeeze(-1)
  # Score tangent-space coverage, not an ill-conditioned inverse of finite motion.
  # Exact FK differences below are used for physical spring magnitude/slew.
  estimate=(A.transpose(1,2)@torch.linalg.solve(A@A.transpose(1,2)+1e-5*torch.eye(3*k,device=J.device),y[...,None])).squeeze(-1)
  error=(estimate-delta).square();energy=delta.square();ratio=error.sum(-1)/(energy.sum(-1)+1e-8)
  # Ignore near-static groups, but retain overall residual and cross-talk errors.
  group_ids=[slice(0,3),slice(3,6),slice(6,9),slice(9,16),slice(16,23)]
  ratios=[]
  for group in group_ids:
   eg=energy[:,group].sum(-1);ratios.append(torch.where((eg>.0025)&(eg>.1*energy.sum(-1)),error[:,group].sum(-1)/(eg+1e-8),0.))
  group_error=torch.stack(ratios,-1).amax(-1)
  force=80*(P1-P)[rows[:,None],ids];next_force=80*(P2-P1)[rows[:,None],ids]
  norms=force.norm(dim=-1);total=norms.sum(-1);point=norms.amax(-1);rate=(next_force-force).norm(dim=-1).amax(-1)/.02
  static=energy.sum(-1)<1e-4
  # At least three/five eligible above-floor candidates must exist.
  geometry_ok=valid.sum(-1)>=k
  good=(ratio<=thresholds['residual'])&(group_error<=thresholds['group_residual'])&geometry_ok
  good|=static&geometry_ok
  budget=(point<=30)&(total<=60)
  result[k]=torch.stack((good.float(),ratio,group_error,total,point,rate,budget.float(),static.float()),-1).cpu().numpy()
 return result

def run(a,name,geom):
 manifest_path=a.stage1/f'{name}.allowlist.json';manifest=json.loads(manifest_path.read_text());mapping={c['clip_id']:c for c in manifest['clips']}
 path=Path(manifest['source']);before=manifest['source_identity']
 for p,v in before.items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v
 it,total,fmt=source(path,50);out={3:[],5:[]};counts={3:Counter(),5:Counter()};tags={3:Counter(),5:Counter()};nwin=0;pending=[];input_arrays=[];started=time.time()
 audit=(a.output/f'{name}.scores.jsonl').open('w')
 def flush():
  nonlocal nwin
  if not pending:return
  arrays=np.stack(input_arrays);b,h,_,_=arrays.shape;flat=arrays.reshape(-1,4,36);t=torch.as_tensor(flat,device=a.device)
  scores=score(geom,t[:,0],t[:,1],t[:,2],t[:,3],a.thresholds)
  reshaped={k:v.reshape(b,h,8) for k,v in scores.items()}
  for bi,info in enumerate(pending):
   alternatives=[]
   for number,vv in reshaped.items():
    v=vv[bi];good_fraction=float(v[:,0].mean());budget_fraction=float(v[:,6].mean());rate=float(np.quantile(v[:,5],.95));still='still' in info['window']['tags']
    reasons=[]
    if not still and good_fraction<a.thresholds['pass_fraction']:reasons.append('sparse_expression')
    if budget_fraction<a.thresholds['budget_fraction']:reasons.append('force_budget')
    if rate>a.thresholds['rate_p95']:reasons.append('force_slew')
    alternatives.append(dict(selected_count=number,accepted=not reasons,reasons=reasons,pass_fraction=good_fraction,residual_mean=float(v[:,1].mean()),group_residual_p95=float(np.quantile(v[:,2],.95)),total_force_max_N=float(v[:,3].max()),point_force_max_N=float(v[:,4].max()),force_rate_p95_N_s=rate))
   for k in (3,5):
    choices=alternatives[:k]
    # Keep a passing smaller contact count if a larger one violates force limits.
    best=min(choices,key=lambda x:(not x['accepted'],x['selected_count'] if x['accepted'] else x['residual_mean']))
    counts[k]['windows']+=1;counts[k].update(best['reasons'])
    record={**info,'max_contacts':k,**best};audit.write(json.dumps(record)+'\n')
    if best['accepted']:
     counts[k]['allowed_windows']+=1;tags[k].update(info['window']['tags']);out[k].append(info)
  nwin+=len(pending);pending.clear();input_arrays.clear()
 for i,source_name,q in it:
  if i not in mapping:continue
  for w in mapping[i]['windows']:
   # Eight evenly spaced interior samples, 100ms reconstruction horizon.
   times=np.rint(np.linspace(w['start'],w['end']-6,8)).astype(int)
   sample=q[np.stack((times,times+5,times+1,times+2),-1)]
   pending.append(dict(clip_id=i,source_name=source_name,window=w));input_arrays.append(sample)
   if len(pending)>=a.batch:flush()
  if (i+1)%10000==0:print(name,i+1,'windows',nwin,'seconds',round(time.time()-started),flush=True)
 flush();audit.close();summaries=[]
 for k in (3,5):
  clips={}
  for row in out[k]:
   i=row['clip_id'];c=clips.setdefault(i,dict(clip_id=i,source_name=row['source_name'],windows=[]));c['windows'].append(row['window'])
  frames=0
  for c in clips.values():c['intervals']=merge([(w['start'],w['end']) for w in c['windows']]);frames+=sum(e-s for s,e in c['intervals'])
  summary=dict(dataset=name,max_contacts=k,**dict(counts[k]),allowed_clips=len(clips),allowed_union_hours=frames/50/3600,stage1_union_hours=manifest['summary']['allowed_union_hours'],accepted_tags=dict(tags[k]))
  payload={**manifest,'schema':'compliance-window-allowlist-stage2-v1','stage1_manifest':str(manifest_path.resolve()),'stage1_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'screening':a.rules,'summary':summary,'clips':list(clips.values())}
  (a.output/f'{name}.max{k}.allowlist.json').write_text(json.dumps(payload,indent=2)+'\n');summaries.append(summary);print(json.dumps(summary),flush=True)
 for p,v in before.items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v
 return summaries

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage1',type=Path,default=ROOT/'artifacts/compliance_dataset/20261115_common_quality_v1');p.add_argument('--output',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['boneseed','lafan','amass']);p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=128);p.add_argument('--thresholds',type=Path)
 a=p.parse_args();override=json.loads(a.thresholds.read_text()) if a.thresholds else {};torch.set_num_threads(2);a.thresholds=dict(residual=.65,group_residual=.90,pass_fraction=.80,budget_fraction=.95,rate_p95=150.)
 assert not set(override)-set(a.thresholds);a.thresholds.update(override)
 a.output.mkdir(parents=True,exist_ok=False);geom=Geometry(a.device)
 a.rules=dict(thresholds=a.thresholds,candidates=geom.points,normalized_scales=geom.scale.cpu().tolist(),groups=['root_translation','root_rotation','waist','left_arm','right_arm'],leg_joints_excluded=True,samples_per_window=8,reconstruction_horizon_frames=5,spring_horizon_frames=1,fps=50,spring_k=80.,selection='greedy per-sample oracle; contacts may change at each sample; no temporal contact schedule optimized',geometry_sha256=hashlib.sha256((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_bytes()).hexdigest(),limitations='Linear tangent-space oracle proxy only (not inversion of exact finite-displacement forces); significant groups require >10% total scaled energy. Geometric proxy only; no learned decoder, dynamics, accessibility, closed-loop perturbations, or contact-switch force continuity. Fixed sampled atlas is not continuous optimization. Test split must remain stage1/unfiltered by this method.')
 (a.output/'RULES.json').write_text(json.dumps(a.rules,indent=2)+'\n');start=time.time();summaries=[]
 for name in a.datasets:summaries+=run(a,name,geom)
 (a.output/'summary.json').write_text(json.dumps(dict(seconds=time.time()-start,results=summaries,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
 lines=['# Stage 2 geometric sparse-contact screening','','| Dataset | Max points | Allowed / stage1 windows | Union hours / stage1 hours |','|---|---:|---:|---:|']
 for s in summaries:lines.append(f"| {s['dataset']} | {s['max_contacts']} | {s.get('allowed_windows',0)} / {s['windows']} | {s['allowed_union_hours']:.2f} / {s['stage1_union_hours']:.2f} |")
 lines+=['','Heuristic sparse geometric screening, NOT learned force-information evaluation or demonstrated tracking. Eight samples per window; nominal reference only. See RULES.json and score audit. Do not use the method-filtered list as a common test set. Source files were unchanged.']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
