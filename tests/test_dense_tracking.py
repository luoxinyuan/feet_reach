import unittest
import torch
from active_adaptation.utils.dense_tracking import component_tracking

class DenseTrackingTests(unittest.TestCase):
    def test_exact_target_and_monotonic_error(self):
        errors=torch.tensor([[0.,0.],[.05,.05],[.2,.2],[1.,1.]])
        scores=component_tracking(errors,[.4,.1]).flatten()
        self.assertEqual(scores[0].item(),1.)
        self.assertTrue((scores[:-1]>scores[1:]).all())
        self.assertTrue(((scores>0)&(scores<=1)).all())
    def test_gradient_moves_each_joint_toward_target(self):
        actual=torch.tensor([[.2,-.1,.8]],requires_grad=True)
        value=component_tracking(actual,[.4,.1]);value.sum().backward()
        self.assertTrue((actual.grad*actual<0).all())
        self.assertTrue(torch.isfinite(actual.grad).all())
    def test_batch_is_independent_and_permutation_invariant(self):
        error=torch.tensor([[.1,.3],[.2,.6]])
        torch.testing.assert_close(component_tracking(error,[.2]),component_tracking(error.flip(-1),[.2]))
        torch.testing.assert_close(component_tracking(error,[.2])[0],component_tracking(error[:1],[.2])[0])
if __name__=='__main__':unittest.main()
