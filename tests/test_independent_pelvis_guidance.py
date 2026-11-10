import unittest
import torch
from active_adaptation.utils.independent_pelvis_guidance import allocate_independent

class IndependentPelvisTest(unittest.TestCase):
 def test_translation_survives_without_joint_motion(self):
  f,g,_,_=allocate_independent(torch.tensor([[.01,0.,0.]]),torch.zeros(1),torch.zeros(1,2,3,4),torch.zeros(1,3),torch.zeros(1,2,3))
  self.assertGreater(float(f[0,0,0]),10);self.assertEqual(float(f[:,1:].abs().sum()),0);self.assertEqual(float(g.sum()),0)
 def test_stationary_root_does_not_cancel_hand_prompt(self):
  jac=torch.zeros(1,2,3,2);jac[0,0,0,1]=1.;jac[0,0,0,0]=10
  f,g,_,_=allocate_independent(torch.zeros(1,3),torch.zeros(1),jac,torch.ones(1,1),torch.zeros(1,2,3))
  self.assertEqual(float(f[0,0].abs().sum()),0);self.assertGreater(float(f[0,1,0]),.8);self.assertEqual(float(f[0,2].abs().sum()),0)
 def test_caps_and_independent_pelvis(self):
  torch.manual_seed(1);n=100
  dp=torch.randn(n,3);yaw=torch.randn(n);jac=torch.randn(n,2,3,18)
  f,gate,_,_=allocate_independent(dp,yaw,jac,100*torch.randn(n,17),torch.randn(n,2,3))
  f2,_,_,_=allocate_independent(dp,yaw,jac,torch.zeros(n,17),torch.zeros(n,2,3))
  torch.testing.assert_close(f[:,0],f2[:,0]);self.assertLessEqual(float(f.norm(dim=-1).max()),30.00001);self.assertLessEqual(float(f.norm(dim=-1).sum(-1).max()),60.00001)
if __name__=='__main__':unittest.main()
