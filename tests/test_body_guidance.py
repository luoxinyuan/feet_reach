import torch
from active_adaptation.utils.body_guidance import allocate_forces,limit_force,slew_force


def test_force_and_yaw_intent():
    arms=torch.tensor([[[0.,.3,.4],[0.,-.3,.4],[.2,0.,.5]]])
    zero=torch.zeros_like(arms);active=torch.ones(1,3,dtype=torch.bool)
    for fd,td in [(torch.tensor([[8.,0.,0.]]),torch.zeros(1)),(torch.zeros(1,3),torch.tensor([2.]))]:
        f=allocate_forces(zero,arms,fd,td,active,zero)
        if td.item():assert (torch.cross(arms,f,dim=-1).sum(1)[:,2]*td).item()>0
        else:assert (f.sum(1)*fd).sum().item()>0


def test_force_budgets_and_slew():
    torch.manual_seed(0)
    desired=limit_force(torch.randn(100,3,3)*100)
    assert desired.norm(dim=-1).max()<30.0001
    assert desired.norm(dim=-1).sum(-1).max()<60.0001
    prev=limit_force(torch.randn(100,3,3)*50)
    f=slew_force(prev,desired,.005)
    assert (f-prev).norm(dim=-1).max()<.7501
    assert f.norm(dim=-1).max()<30.0001
    assert f.norm(dim=-1).sum(-1).max()<60.0001


def test_inactive_and_degenerate_points():
    zero=torch.zeros(3,3,3);active=torch.tensor([[True,False,False],[True,True,False],[True,True,True]])
    f=allocate_forces(torch.ones_like(zero),zero,torch.ones(3,3),torch.ones(3),active,zero)
    assert torch.isfinite(f).all()
    assert f[~active].count_nonzero()==0


def test_nonopposing_global_intent():
    from active_adaptation.utils.body_guidance import consistent_force
    torch.manual_seed(23)
    f=torch.randn(256,3,3)*30;arms=torch.randn_like(f)*.3
    fd=torch.randn(256,3)*10;td=torch.randn(256)*2
    active=torch.rand(256,3)>.3;f*=active[...,None]
    f=limit_force(consistent_force(f,arms,fd,td,active))
    assert ((f.sum(1)*fd).sum(-1)>=-1e-4).all()
    assert ((torch.cross(arms,f,dim=-1).sum(1)[:,2]*td)>=-1e-4).all()
    assert f[~active].count_nonzero()==0
