"""Continuous mesh contacts, batched force fitting and triangle optimization."""
import json
from pathlib import Path
import numpy as np
import torch
from .body_guidance import limit_force
from .jacobian_guidance import point_linear_jacobian

def quat_apply(q,v):
    # Unit quaternions in wxyz order, matching Isaac's body-frame convention.
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)

class SurfaceSampler:
    def __init__(self,path,body_names,device):
        path=Path(path);self.manifest=json.loads((path/'manifest.json').read_text())
        raw=np.load(path/'triangles.npy');parts=self.manifest['parts'];tri=[];body=[]
        for p in parts:
            if p['body'] not in body_names:raise ValueError(f"Surface body missing: {p['body']}")
            tri.append(raw[p['start']:p['end']]);body.extend([body_names.index(p['body'])]*(p['end']-p['start']))
        self.triangles=torch.tensor(np.concatenate(tri),device=device);self.body=torch.tensor(body,device=device)
        v=self.triangles;area=torch.cross(v[:,1]-v[:,0],v[:,2]-v[:,0],dim=-1).norm(dim=-1)*.5
        # Uniform across exposed links, area-uniform within each link.
        totals=torch.zeros(len(body_names),device=device).scatter_add_(0,self.body,area)
        prob=area/totals[self.body];self.cdf=prob.cumsum(0);self.cdf/=self.cdf[-1].clone();self.device=device
    def sample(self,shape):
        ids=torch.searchsorted(self.cdf,torch.rand(shape,device=self.device)).clamp_max(len(self.body)-1)
        u=torch.rand(*shape,2,device=self.device);s=u[...,0].sqrt();bary=torch.stack([1-s,s*(1-u[...,1]),s*u[...,1]],-1)
        return self.body[ids],self.triangles[ids],bary


def geometry(raw,body_pos,body_com,body_quat,body,triangle,bary):
    rows=torch.arange(len(body),device=body.device)[:,None]
    local=(triangle*bary[...,None]).sum(-2)
    offset=quat_apply(body_quat[rows,body],local)
    world=body_pos[rows,body]+offset
    jac=point_linear_jacobian(raw[rows,body],world-body_com[rows,body])
    return local,world,jac


def solve(jac,target,weights,active,previous,force_penalty=1.,smooth_penalty=1.):
    n,k,_,d=jac.shape;mask=active[...,None].expand(n,k,3).reshape(n,k*3)
    A=jac.reshape(n,k*3,d).transpose(1,2)*mask[:,None]
    Aw=A*weights[:,:,None];b=target*weights
    H=Aw.transpose(1,2)@Aw+(force_penalty+smooth_penalty)*torch.eye(k*3,device=A.device)
    rhs=(Aw.transpose(1,2)@b[...,None]).squeeze(-1)+smooth_penalty*previous.flatten(1)*mask
    force=torch.linalg.solve(H,rhs[...,None]).squeeze(-1).reshape(n,k,3)*active[...,None]
    return limit_force(force)


def objective(jac,targets,weights,active,contact_penalty=.025):
    prev=torch.zeros(*active.shape,3,device=jac.device);scores=[]
    for target,w in zip(targets.unbind(1),weights.unbind(1)):
        f=solve(jac,target,w,active,prev);error=((torch.einsum('nkcd,nkc->nd',jac,f)-target)*w).square().sum(-1)
        scores.append(error/((target*w).square().sum(-1)+.01));prev=f
    return torch.stack(scores,-1).mean(-1)+contact_penalty*active.sum(-1)


def project_simplex(x):
    # Exact Euclidean projection of barycentric triples onto the triangle.
    u=x.sort(dim=-1,descending=True).values;css=u.cumsum(-1)-1
    j=torch.arange(1,4,device=x.device,dtype=x.dtype);cond=u-css/j>0
    rho=cond.sum(-1,keepdim=True).clamp_min(1)-1
    theta=css.gather(-1,rho)/(rho+1);return (x-theta).clamp_min(0)
