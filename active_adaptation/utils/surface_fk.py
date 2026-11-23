"""Level-batched full-body FK from exported training USD joint frames."""
import torch


def quat_apply(q,v):
    # Unit quaternions in wxyz order, matching Isaac's body-frame convention.
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)

def multiply(a,b):
 return torch.cat((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True),a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)),-1)

class SurfaceFK:
 def __init__(self,geometry,joint_names,device):
  self.bodies=['pelvis']+[e['child'] for e in geometry['edges']];pending=list(geometry['edges']);available={'pelvis'};self.levels=[]
  while pending:
   ready=[e for e in pending if e['parent'] in available]
   if not ready:raise ValueError('Disconnected geometry')
   tensor=lambda x,dtype=torch.float32:torch.tensor(x,device=device,dtype=dtype)
   ids=tensor([self.bodies.index(e['child']) for e in ready],torch.long);parents=tensor([self.bodies.index(e['parent']) for e in ready],torch.long)
   joint=tensor([joint_names.index(e['name']) if e['axis'] else 0 for e in ready],torch.long)
   axes=tensor([[int(e['axis']==c) for c in 'XYZ'] for e in ready]);p0=tensor([e['p0'] for e in ready]);p1=tensor([e['p1'] for e in ready]);r0=tensor([e['r0'] for e in ready]);r1=tensor([e['r1'] for e in ready]);r1[:,1:]*=-1
   self.levels.append((ids,parents,joint,axes,p0,p1,r0,r1))
   for e in ready:available.add(e['child']);pending.remove(e)
 def __call__(self,q,rp,rq):
  n=len(q);P=q.new_zeros(n,len(self.bodies),3);R=q.new_zeros(n,len(self.bodies),4);P[:,0]=rp;R[:,0]=rq
  for ids,parents,joint,axes,p0,p1,r0,r1 in self.levels:
   angle=q[:,joint,None]*axes.norm(dim=-1)[None,:,None]/2
   jr=torch.cat((angle.cos(),angle.sin()*axes),-1)
   cr=multiply(multiply(multiply(R[:,parents],r0.expand(n,-1,-1)),jr),r1.expand(n,-1,-1))
   P[:,ids]=P[:,parents]+quat_apply(R[:,parents],p0.expand(n,-1,-1))-quat_apply(cr,p1.expand(n,-1,-1));R[:,ids]=cr
  return P,R
