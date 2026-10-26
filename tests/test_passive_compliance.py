import unittest
import torch
from active_adaptation.utils.passive_compliance import fit_spring, spring_force


class PassiveForceTests(unittest.TestCase):
    def test_force_direction_damping_cap_and_hold(self):
        position=torch.zeros(3,2,3);anchor=torch.ones_like(position)
        velocity=torch.zeros_like(position);active=torch.ones(3,2,dtype=torch.bool)
        stiffness=torch.full((3,2),80.)
        for limit in (30.,60.):
            force=spring_force(anchor,position,velocity,stiffness,active,limit=limit)
            torch.testing.assert_close(force.norm(dim=-1),torch.full((3,2),limit))
            self.assertTrue((force>0).all())
        force=spring_force(position,position,torch.ones_like(position),stiffness,active)
        self.assertTrue((force<0).all())
        active.zero_()
        self.assertEqual(spring_force(anchor,position,velocity,stiffness,active).count_nonzero(),0)

    def test_fit_is_bounded_and_reduces_objective(self):
        torch.manual_seed(1)
        path=torch.randn(5,12,2,3)*.03
        dt=.05; b=8.;k=fit_spring(path,dt,b)
        self.assertTrue(((k>=40)&(k<=160)).all())
        velocity=(path[:,2:]-path[:,:-2])/(2*dt)
        acceleration=(path[:,2:]-2*path[:,1:-1]+path[:,:-2])/dt**2
        error=path[:,-1:]-path[:,1:-1]
        def objective(gain):
            return (gain[:,None,:,None]*error-b*velocity-acceleration).square().sum((1,3))+.01*(gain-80).square()
        self.assertTrue((objective(k)<=objective(torch.full_like(k,80.))+1e-3).all())
        self.assertTrue(torch.isfinite(fit_spring(torch.zeros_like(path),dt)).all())

if __name__=='__main__':unittest.main()
