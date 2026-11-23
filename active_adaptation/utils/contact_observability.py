"""Continuous-contact selection by local motion observability (no dynamics).

A contact reports a Cartesian displacement J dx, via F=k J dx. Select rows
of J that best reconstruct the intended generalized displacement. The
ridge reconstruction, not force magnitude, scores a contact set.
"""
import torch


def reconstruction_score(jac, delta, ridge=.002, covariance=None):
    # jac [B,K,3,D], delta [B,H,D]; isotropic joint/root increments in native units.
    A=jac.flatten(1,2)
    y=A@delta.transpose(1,2)
    CA=A.transpose(1,2) if covariance is None else covariance@A.transpose(1,2)
    c=torch.linalg.solve(A@CA+ridge*torch.eye(A.shape[1],device=A.device), y)
    estimate=(CA@c).transpose(1,2)
    return ((estimate-delta).square().sum(-1)/(delta.square().sum(-1)+1e-7)).mean(-1)


def greedy_contacts(jac, delta, maximum, penalty=.015, covariance=None, coverage=None, coverage_penalty=0.):
    """Return ordered continuous candidate IDs and an economical prefix mask.

    Every call uses freshly sampled triangles/barycentric locations. Candidate
    IDs are temporary numerical search samples, not a fixed contact atlas.
    """
    B,P,_,D=jac.shape;rows=torch.arange(B,device=jac.device)
    chosen=[];best=torch.full((B,),float('inf'),device=jac.device)
    count=torch.ones(B,dtype=torch.long,device=jac.device)
    for k in range(maximum):
        candidate=jac[:,:,None] if not chosen else torch.cat((jac[rows[:,None],torch.stack(chosen,1)][:,None].expand(-1,P,-1,-1,-1),jac[:,:,None]),2)
        score=reconstruction_score(candidate.reshape(B*P,k+1,3,D),delta[:,None].expand(-1,P,-1,-1).reshape(B*P,*delta.shape[1:]),covariance=covariance).reshape(B,P)
        if coverage is not None:
            covered=coverage if not chosen else coverage | coverage[rows[:,None],torch.stack(chosen,1)].any(1)[:,None]
            score=score+coverage_penalty*(~covered).float().sum(-1)
        score=score.masked_fill(jac.square().sum((-1,-2))<1e-12,float('inf'))
        if chosen:score.scatter_(1,torch.stack(chosen,1),float('inf'))
        index=score.argmin(-1);value=score[rows,index]+penalty*(k+1)
        count=torch.where(value<best,k+1,count);best=torch.minimum(best,value);chosen.append(index)
    return torch.stack(chosen,1),torch.arange(maximum,device=jac.device)[None]<count[:,None]
