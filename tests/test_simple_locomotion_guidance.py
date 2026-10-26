import unittest
import torch
from active_adaptation.utils.simple_locomotion_guidance import guidance

class GuidanceTests(unittest.TestCase):
 def test_pull_twist_limits_and_hold(self):
  p=torch.zeros(3,3);q=torch.tensor([[1.,0.,0.,0.]]).repeat(3,1)
  anchor=torch.tensor([[.2,0.,5.],[5.,5.,2.],[5.,0.,0.]])
  yaw=torch.tensor([.35,2.,1.]);hold=torch.tensor([False,False,True])
  f,t=guidance(p,q,anchor,yaw,hold)
  torch.testing.assert_close(f[0],torch.tensor([12.,0.,0.]))
  self.assertAlmostEqual(t[0,2].item(),2.8,places=5)
  self.assertLessEqual(f.norm(dim=-1).max(),30.00001);self.assertLessEqual(t.abs().max(),4.)
  self.assertEqual(f[2].count_nonzero(),0);self.assertEqual(t[2].count_nonzero(),0)
  self.assertEqual(f[:,2].count_nonzero(),0)
 def test_yaw_wrap_uses_shortest_turn(self):
  yaw=torch.tensor([torch.pi-.01]);q=torch.stack([torch.cos(yaw/2),yaw*0,yaw*0,torch.sin(yaw/2)],-1)
  f,t=guidance(torch.zeros(1,3),q,torch.zeros(1,3),torch.tensor([-torch.pi+.01]),torch.tensor([False]))
  self.assertAlmostEqual(t[0,2].item(),.16,places=4)

if __name__=='__main__':unittest.main()
