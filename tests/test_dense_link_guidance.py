"""Invariants of dense force construction, independent of Isaac and learned models."""
import unittest
import numpy as np
from active_adaptation.utils.dense_link_guidance import displacement, cap_vectors, unit_metrics, choose_k

class DenseForcesTest(unittest.TestCase):
    def test_clip_boundary_and_translation(self):
        x=np.array([[[0.,1,0]],[[1,2,0]],[[999,0,0]],[[1000,1,0]]],dtype=np.float32)
        valid=np.array([True,False,True]);delta=displacement(x,valid)
        np.testing.assert_array_equal(delta[1],0)
        np.testing.assert_allclose(delta,displacement(x+[10,-4,3],valid))
        np.testing.assert_allclose(delta[[0,2]],[[[1,1,0]],[[1,1,0]]])

    def test_fixed_gain_accounts_for_dense_sum(self):
        delta=np.ones((100,23,3),dtype=np.float32)*[.02,0,0]
        delta[-1]*=5
        m=unit_metrics(delta,np.zeros_like(delta));k,limits=choose_k(m)
        f=delta*k
        self.assertLess(k,limits['point_max'])
        self.assertLessEqual(np.linalg.norm(f,axis=-1).max(),30+1e-5)
        self.assertLessEqual(np.linalg.norm(f.sum(1),axis=-1).max(),80+1e-5)
        self.assertLessEqual(np.linalg.norm(f,axis=-1).sum(-1).max(),120+1e-5)

    def test_rotational_equivariance_and_clip_direction(self):
        rng=np.random.default_rng(42);f=rng.normal(size=(12,23,3))*20
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(cap_vectors(f@rotation.T),cap_vectors(f)@rotation.T,atol=1e-6)
        self.assertLessEqual(np.linalg.norm(cap_vectors(f),axis=-1).max(),30+1e-6)
        np.testing.assert_allclose(np.cross(f,cap_vectors(f)),0,atol=1e-10)
if __name__=='__main__':unittest.main()
