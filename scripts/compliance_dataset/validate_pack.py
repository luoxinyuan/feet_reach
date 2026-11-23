#!/usr/bin/env python3
import argparse,json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active_adaptation.utils.motion import MotionDataset,G1_JOINT_NAMES
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device="cuda:0").app
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset
from pack import normalized_qpos,derivatives
from scan import source

p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'dataset/compliance_mix_29h_v1');a=p.parse_args();torch.set_num_threads(2)
m=json.loads((a.dataset/'manifest.json').read_text());cfg=json.loads((a.dataset/'dataset_config.json').read_text());index=json.loads(Path(m['source_allowlist']).read_text());sources={k:json.loads(Path(v).read_text()) for k,v in index['datasets'].items()};rng=np.random.default_rng(20261115)
sp=importlib.util.spec_from_file_location('independent_geometry',ROOT/'active_adaptation/utils/segmented_guidance.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
geom=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/geometry.json').read_text());total=0;clips=0;max_fk=0.;max_norm=0.;max_floor=0.;queries={};datasets={};category_report={}
for category in m['categories']:
 path=a.dataset/'train'/category;ds=MotionDataset.create_from_path_lazy(str(path));datasets[category]=ds;assert ds.joint_names==G1_JOINT_NAMES
 prov=json.loads((path/'provenance.json').read_text());starts=ds.starts.numpy();ends=ds.ends.numpy();assert starts[0]==0 and np.array_equal(starts[1:],ends[:-1]);assert ends[-1]==len(ds.data);assert len(prov)==len(starts)
 for key in ds.data.__dataclass_fields__:
  values=getattr(ds.data,key)
  for chunk in values.split(65536):assert torch.isfinite(chunk).all(),(category,key)
 for i,row in enumerate(prov):
  assert row['motion_id']==i and row['start']==int(starts[i]) and row['end']==int(ends[i])
  n=int(ends[i]-starts[i]);assert torch.equal(ds.data.step[starts[i]:ends[i]],torch.arange(n,dtype=torch.int32));assert (ds.data.motion_id[starts[i]:ends[i]]==i).all()
 ankles=[ds.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
 height=ds.data.body_pos_w[:,ankles,2].numpy().min(-1);mins=np.minimum.reduceat(height,starts);max_floor=max(max_floor,float(abs(mins).max()));assert abs(mins).max()<1e-5
 norm=(ds.data.root_quat_w.norm(dim=-1)-1).abs().max().item();max_norm=max(max_norm,norm);assert norm<1e-5
 ids=rng.integers(0,len(ds.data),64);d=ds.data[torch.tensor(ids)]
 points=[dict(body=b,offset=[0,0,0]) for b in ds.body_names]
 P,_=module.geometry_jacobians(d.joint_pos.numpy(),d.root_pos_w.numpy(),d.root_quat_w.numpy(),ds.joint_names,points,geom)
 error=float(abs(P-d.body_pos_w.numpy()).max());max_fk=max(max_fk,error);assert error<1e-5
 for mid in rng.choice(len(prov),min(16,len(prov)),replace=False):
  row=prov[int(mid)];queries.setdefault((row['dataset'],row['source_clip_id']),[]).append((category,row))
 total+=len(ds.data);clips+=len(starts);category_report[category]=dict(frames=len(ds.data),clips=len(starts),max_independent_fk_error_m=error)
assert total==m['frames'] and clips==m['clips']
source_error=0.
for name,manifest in sources.items():
 it,_,_=source(Path(manifest['source']),50)
 for clip_id,_,raw in it:
  for category,row in queries.get((name,clip_id),[]):
   expected=normalized_qpos(raw[row['source_start']:row['source_end']]);expected[:,2]+=row['translation_applied_xyz'][2]
   data=datasets[category].data[row['start']:row['end']];actual=np.concatenate((data.root_pos_w.numpy(),data.root_quat_w.numpy(),data.joint_pos.numpy()),-1)
   error=float(abs(actual-expected).max());source_error=max(source_error,error);assert error<1e-6
   lv,av,jv=derivatives(expected)
   for actual_v,want in [(data.root_lin_vel_w,lv),(data.root_ang_vel_w,av),(data.joint_vel,jv)]:assert np.allclose(actual_v.numpy(),want,atol=1e-5)
 for path,stat in manifest['source_identity'].items():assert [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==stat
# Exercise the exact production loader, including on-reset category resampling.
torch.manual_seed(20261115);extra=[dict(name='primary_id',shape=1,dtype=torch.long)]
loader=ProgressiveMultiMotionDataset(**cfg,env_size=256,max_step_size=200,dataset_extra_keys=extra,device=torch.device('cpu'))
counts=torch.zeros(8,dtype=torch.long)
for _ in range(30):
 lengths=loader.reset(torch.arange(256));assert (lengths>=50).all() and (lengths<=200).all()
 info=loader.get_slice_info(torch.arange(256))['primary_id'].flatten();counts+=torch.bincount(info,minlength=8)
 first=loader.get_slice(None,torch.zeros(256,dtype=torch.long),1);last=loader.get_slice(None,torch.full((256,),9999,dtype=torch.long),1)
 assert torch.isfinite(first.joint_pos).all() and torch.isfinite(last.joint_pos).all()
 assert ((last.step[:,0]+1)==lengths).all()
prob=counts.float()/counts.sum();expected=torch.tensor(cfg['path_weights']);assert (prob-expected).abs().max()<.025
result=dict(passed=True,frames=total,clips=clips,hours=total/180000,categories=category_report,max_independent_fk_error_m=max_fk,max_root_quaternion_norm_error=max_norm,max_floor_alignment_error_m=max_floor,max_source_pose_error=source_error,source_pose_and_derivative_checks=128,loader_draws=int(counts.sum()),expected_reset_probabilities=expected.tolist(),observed_reset_probabilities=prob.tolist(),source_files_unchanged=True,clip_boundaries_and_end_clamping=True)
(a.dataset/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

app.close()
