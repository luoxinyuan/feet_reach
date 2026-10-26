import torch
from active_adaptation.utils.point_guidance import reduce_point_forces


def test_same_link_couple_is_preserved():
    pos=torch.tensor([[[1.,0.,0.],[-1.,0.,0.],[0.,0.,0.]]])
    f=torch.tensor([[[0.,2.,0.],[0.,-2.,0.],[0.,0.,0.]]])
    ids=torch.zeros(1,3,dtype=torch.long)
    force,moment=reduce_point_forces(f,pos,ids,torch.zeros(1,2,3))
    assert force.count_nonzero()==0
    torch.testing.assert_close(moment[0,0],torch.tensor([0.,0.,4.]))
    assert moment[0,1].count_nonzero()==0


def test_net_wrench_and_power_equivalence():
    torch.manual_seed(19)
    f=torch.randn(100,3,3);pos=torch.randn_like(f);bpos=torch.randn(100,5,3)
    ids=torch.randint(0,5,(100,3))
    force,moment=reduce_point_forces(f,pos,ids,bpos)
    torch.testing.assert_close(force.sum(1),f.sum(1))
    torch.testing.assert_close((moment+torch.cross(bpos,force,dim=-1)).sum(1),torch.cross(pos,f,dim=-1).sum(1),atol=2e-6,rtol=1e-5)
    v=torch.randn_like(bpos);w=torch.randn_like(bpos);i=ids[...,None].expand_as(f)
    pv=v.gather(1,i)+torch.cross(w.gather(1,i),pos-bpos.gather(1,i),dim=-1)
    torch.testing.assert_close((f*pv).sum((1,2)),(force*v+moment*w).sum((1,2)),atol=3e-6,rtol=1e-5)
