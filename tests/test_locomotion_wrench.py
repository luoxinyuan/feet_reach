import sys,unittest
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'active_adaptation/utils'))
from locomotion_wrench import skew,grasp_matrix,feasible_yaw_wrench,allocate_contact_wrench,guidance_wrench

class WrenchTests(unittest.TestCase):
    def test_rank_and_missing_moment(self):
        points=np.array([[.1,.25,.2],[.1,-.25,.2]])
        A=grasp_matrix(points,np.zeros(3));self.assertEqual(np.linalg.matrix_rank(A),5)
        raw=np.array([10.,0.,0.,0.,0.,2.])
        self.assertGreater(np.linalg.norm(A@np.linalg.lstsq(A,raw,rcond=None)[0]-raw),1.)
        desired=feasible_yaw_wrench(points,np.zeros(3),raw[:3],raw[5])
        np.testing.assert_allclose(A@np.linalg.lstsq(A,desired,rcond=None)[0],desired,atol=1e-10)
        np.testing.assert_allclose(desired[:3],raw[:3]);self.assertAlmostEqual(desired[5],raw[5])

    def test_yaw_and_translation_covariance(self):
        points=np.array([[.1,.25,.2],[.15,-.25,.3]]);com=np.zeros(3);force=np.array([5.,2.,0.])
        R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        desired=feasible_yaw_wrench(points,com,force,3.)
        shifted=feasible_yaw_wrench(points@R.T+2,com+2,R@force,3.)
        np.testing.assert_allclose(shifted,np.r_[R@desired[:3],R@desired[3:]],atol=1e-10)

    def test_standing_inverse_dynamics(self):
        hands=np.array([[.1,.25,.2],[.1,-.25,.2]]);feet=np.array([[0.,.12,-.8],[0.,-.12,-.8]])
        def jac(p):
            return np.c_[np.block([[np.eye(3),-skew(p)],[np.zeros((3,3)),np.eye(3)]]),np.zeros(6)]
        h=np.r_[0.,0.,30*9.81,0.,0.,0.,0.]
        target=feasible_yaw_wrench(hands,np.zeros(3),np.array([4.,0.,0.]),1.)
        result=allocate_contact_wrench(h,[jac(p) for p in hands],[jac(p) for p in feet],[True,True],hands,np.zeros(3),target,np.array([20.]))
        self.assertTrue(result['success'],result)
        np.testing.assert_allclose(result['achieved'],target,atol=1e-5)
        self.assertLess(result['equilibrium_residual'],1e-6)
        self.assertLessEqual(np.linalg.norm(result['forces'],axis=-1).max(),60.)
        no_support=allocate_contact_wrench(h,[jac(p) for p in hands],[jac(p) for p in feet],[False,False],hands,np.zeros(3),target,np.array([20.]))
        self.assertFalse(no_support['success'])
        too_fast=allocate_contact_wrench(h,[jac(p) for p in hands],[jac(p) for p in feet],[True,True],hands,np.zeros(3),target,np.array([20.]),previous=np.zeros(6),max_force_rate=1.,time_step=.02)
        self.assertFalse(too_fast['success'])

    def test_constant_motion_has_informative_force(self):
        w=guidance_wrench(np.array([.5,0.,0.]),np.zeros(3),.2,0.)
        self.assertGreater(w[0],0);self.assertGreater(w[5],0)
        np.testing.assert_array_equal(guidance_wrench(np.zeros(3),np.zeros(3),0.,0.),np.zeros(6))

if __name__=='__main__':unittest.main()
