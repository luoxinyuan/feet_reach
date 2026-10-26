"""Fixed expert transition pool from exactly the segmented baseline's train view."""
import json,hashlib
from pathlib import Path
import torch
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.compliance_amp import motion_features,pair_features,AMP_BODIES
root=Path(__file__).resolve().parents[2];torch.set_num_threads(4)
view=root/'artifacts/ceer_segmented_jacobian/train18/dataset'
ds=MotionDataset.create_from_path_lazy(str(view))
g=torch.Generator().manual_seed(20261024);n=100000
clips=torch.randint(ds.num_motions,(n,),generator=g)
idx=ds.starts[clips].long()+(torch.rand(n,generator=g)*(ds.lengths[clips]-1)).long()
body_ids=[ds.body_names.index(n) for n in AMP_BODIES]
features=[]
for indices in (idx,idx+1):
 chunks=[]
 for ii in indices.split(4096):
  m=ds.data[ii];rp=m.root_pos_w.clone();rp[:,2]+=.03540915
  points=m.body_pos_w[:,body_ids].clone();points[:,:,2]+=.03540915
  chunks.append(motion_features(m.joint_pos,m.joint_vel,rp,m.root_quat_w,m.root_lin_vel_w,m.root_ang_vel_w,points))
 features.append(torch.cat(chunks))
pairs=pair_features(*features)
frames=torch.cat(features);mean=frames.mean(0).repeat(2);scale=frames.std(0).clamp_min(.1).repeat(2)
out=root/'artifacts/ceer_segmented_amp';out.mkdir(exist_ok=True)
provenance=dict(expert_view=str(view),meta_sha256=hashlib.sha256((view/'meta_motion.json').read_bytes()).hexdigest(),
    pairs=n,feature_dim=features[0].shape[-1],joint_names=ds.joint_names,body_names=AMP_BODIES,dt=.02,seed=20261024,
    description='uniform accepted training interval, uniform valid consecutive pair; no val data, no cross-reset pairs')
torch.save(dict(pairs=pairs,mean=mean,scale=scale,provenance=provenance),out/'expert_pairs.pt')
(out/'expert_manifest.json').write_text(json.dumps(provenance,indent=2));print(provenance)
