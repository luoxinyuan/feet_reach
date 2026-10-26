import torch
from active_adaptation.utils.compliance_amp import motion_features,pair_features,style_reward

def test_motion_features_global_heading_invariance():
    g=torch.Generator().manual_seed(13);n=8
    q=torch.randn(n,29,generator=g);qd=torch.randn(n,29,generator=g)
    pos=torch.randn(n,3,generator=g);quat=torch.zeros(n,4);quat[:,0]=1
    lin=torch.randn(n,3,generator=g);ang=torch.randn(n,3,generator=g);points=torch.randn(n,5,3,generator=g)
    first=motion_features(q,qd,pos,quat,lin,ang,points)
    angle=.7;c=torch.cos(torch.tensor(angle));s=torch.sin(torch.tensor(angle))
    R=torch.tensor([[c,-s,0],[s,c,0],[0,0,1.]])
    offset=torch.tensor([12.,-4.,0.]);rot=quat.clone();rot[:,0]=torch.cos(torch.tensor(angle/2));rot[:,3]=torch.sin(torch.tensor(angle/2))
    second=motion_features(q,qd,pos@R.T+offset,rot,lin@R.T,ang@R.T,points@R.T+offset)
    torch.testing.assert_close(first,second,atol=3e-6,rtol=1e-5)
    assert first.shape==(n,83) and pair_features(first,second).shape==(n,166)

def test_reward_is_bounded_and_favors_expert():
    scores=torch.tensor([-100.,-1.,0.,1.,100.]);r=style_reward(scores)
    assert r.tolist()==[0.,0.,.75,1.,0.]

if __name__=='__main__':
    test_motion_features_global_heading_invariance();test_reward_is_bounded_and_favors_expert();print('AMP_UNIT_PASS')
