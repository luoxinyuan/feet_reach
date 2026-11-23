"""Physics-independent force, mesh, virtual-work and optimization checks."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from active_adaptation.utils.surface_guidance import SurfaceSampler,geometry,solve,objective,project_simplex
from active_adaptation.utils.body_guidance import slew_force

torch.manual_seed(123);torch.set_num_threads(2)
names=list(dict.fromkeys(p['body'] for p in json.loads((ROOT/'artifacts/surface_compliance/geometry/manifest.json').read_text())['parts']))
s=SurfaceSampler(ROOT/'artifacts/surface_compliance/geometry',names,'cpu');body,tri,bary=s.sample((128,3))
assert torch.isfinite(tri).all();torch.testing.assert_close(bary.sum(-1),torch.ones(128,3));assert bary.min()>=0
n=128;D=35;raw=torch.randn(n,len(names),6,D);bp=torch.randn(n,len(names),3);com=bp+.02*torch.randn_like(bp);quat=torch.zeros(n,len(names),4);quat[...,0]=1
bary.requires_grad_();local,pos,jac=geometry(raw,bp,com,quat,body,tri,bary)
vel=torch.randn(n,D);twist=torch.einsum('nkcd,nd->nkc',raw[torch.arange(n)[:,None],body],vel)
pvel=twist[...,:3]+torch.cross(twist[...,3:],pos-com[torch.arange(n)[:,None],body],dim=-1)
torch.testing.assert_close(torch.einsum('nkcd,nd->nkc',jac,vel),pvel,rtol=2e-5,atol=2e-5)
force=torch.randn(n,3,3);torque=torch.einsum('nkcd,nkc->nd',jac,force)
torch.testing.assert_close((torque*vel).sum(-1),(force*pvel).sum((1,2)),rtol=2e-5,atol=2e-5)
target=torch.randn(n,3,D);weights=torch.ones_like(target);active=torch.rand(n,3)>.3;active[:,0]=True
score=objective(jac,target,weights,active);grad=torch.autograd.grad(score.sum(),bary)[0];assert torch.isfinite(grad).all()
projected=project_simplex(torch.randn(1000,3));assert projected.min()>=0;torch.testing.assert_close(projected.sum(-1),torch.ones(1000),atol=1e-6,rtol=1e-6)
f=solve(jac.detach(),target[:,0]*100,weights[:,0],active,torch.zeros(n,3,3));norm=f.norm(dim=-1);assert norm.max()<=30.0001 and norm.sum(-1).max()<=60.0001;assert not f[~active].any()
g=slew_force(torch.zeros_like(f),f,.005);assert g.norm(dim=-1).max()<=.75001
print('PASS mesh barycentric sampling, Jacobian point shift, virtual work, differentiable objective, simplex, force and slew bounds')
