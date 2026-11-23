import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from active_adaptation.utils.surface_fk import SurfaceFK
from active_adaptation.utils.segmented_guidance import geometry_jacobians
from active_adaptation.utils.surface_guidance import quat_apply
from active_adaptation.utils.supermimic import ReleasedSuperMimic
m=json.load(open(ROOT/'artifacts/offline_force_baseline/train37_budget60/manifest.json'));g=m['geometry'];names=m['joint_names'];fk=SurfaceFK(g,names,'cpu')
torch.manual_seed(11);q=torch.randn(4,len(names))*.3;rp=torch.randn(4,3);rq=torch.randn(4,4);rq/=rq.norm(dim=-1,keepdim=True);P,R=fk(q,rp,rq)
off=torch.randn(len(fk.bodies),3)*.1;points=[dict(body=b,offset=o.tolist()) for b,o in zip(fk.bodies,off)];truth,_=geometry_jacobians(q.numpy(),rp.numpy(),rq.numpy(),names,points,g)
torch.testing.assert_close(P+quat_apply(R,off[None].expand(4,-1,-1)),torch.tensor(truth,dtype=torch.float32),atol=2e-6,rtol=2e-5)
net=ReleasedSuperMimic();obs=torch.randn(4,930);goal=torch.randn(4,680)
torch.testing.assert_close(net(obs,goal),net.decode(obs,net.encode(goal)),atol=0,rtol=0)
assert not any(p.requires_grad for p in net.parameters())
print('PASS: full-body FK including wrists; frozen latent encoder/decoder exact equivalence')
