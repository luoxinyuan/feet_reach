"""Point-cloud gradients and teacher/student transfer in the residual controller."""
import copy
import unittest
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from torchrl.data import Composite, UnboundedContinuous
from active_adaptation.learning.hierarchical.root_student_ppo import RootStudentPPOConfig, RootStudentPPOPolicy
from active_adaptation.learning.modules.pointnet import GeometryPointNet

class ResidualPointNetTest(unittest.TestCase):
    def make_policy(self, points=8):
        cfg=OmegaConf.structured(RootStudentPPOConfig(geometry_points=points))
        spec=Composite(hl_policy=UnboundedContinuous((2,10)),hl_priv=UnboundedContinuous((2,6)),geometry_=UnboundedContinuous((2,32)),shape=[2])
        policy=RootStudentPPOPolicy(cfg,spec,SimpleNamespace(shape=(3,)),None,device='cpu')
        td=spec.zero();cloud=torch.randn(2,8,4);cloud[:,:,3]=1.;td['geometry_']=cloud.flatten(1)
        return policy,td

    def test_actor_pointnet_gradient_and_transfer(self):
        policy,td=self.make_policy();policy.encoder_priv(td);policy.actor_teacher(td)
        td['loc'].square().sum().backward()
        grads=[p.grad.abs().sum().item() for m in policy.actor_teacher.modules() if isinstance(m,GeometryPointNet) for p in m.parameters() if p.grad is not None]
        self.assertGreater(sum(grads),0.)
        teacher=copy.deepcopy(policy.actor_teacher.state_dict());policy._copy_teacher_to_student()
        for key,value in teacher.items():torch.testing.assert_close(policy.actor_student.state_dict()[key],value)
        public=td.select('hl_policy','geometry_');policy.get_rollout_policy('deploy')(public)
        self.assertEqual(public['action'].shape,(2,3))
        self.assertTrue(torch.isfinite(public['action']).all())

    def test_permutation_and_empty_geometry(self):
        policy,td=self.make_policy();policy.encoder_priv(td);policy.actor_teacher(td);expected=td['loc'].clone()
        td['geometry_']=td['geometry_'].reshape(2,8,4)[:,torch.randperm(8)].flatten(1)
        policy.actor_teacher(td);torch.testing.assert_close(td['loc'],expected)
        td['geometry_'].zero_();policy.actor_teacher(td);policy.critic(td)
        self.assertTrue(torch.isfinite(td['loc']).all());self.assertTrue(torch.isfinite(td['state_value']).all())

    def test_legacy_without_geometry(self):
        policy,td=self.make_policy(points=0)
        self.assertFalse(any(isinstance(m,GeometryPointNet) for m in policy.modules()))
        public=td.select('hl_policy');policy.get_rollout_policy('deploy')(public)
        self.assertEqual(public['action'].shape,(2,3))

if __name__=='__main__':unittest.main()
