import unittest
import torch
from active_adaptation.utils.foot_force import FootForceRamp


class FootForceTests(unittest.TestCase):
    def test_linear_cycle_bound_and_reset(self):
        torch.manual_seed(12)
        r=FootForceRamp(128,'cpu',ramp_up_range=(2,2),hold_range=(2,2),
                        ramp_down_range=(2,2),rest_range=(2,2),zero_prob=0)
        f=[r.advance().clone() for _ in range(8)]
        self.assertEqual(f[0].count_nonzero(),0)
        self.assertEqual(f[1].count_nonzero(),0)
        torch.testing.assert_close(f[2]*2,f[3])
        torch.testing.assert_close(f[3],f[4])
        torch.testing.assert_close(f[4],f[5])
        torch.testing.assert_close(f[6]*2,f[5])
        self.assertEqual(f[7].count_nonzero(),0)
        self.assertLessEqual(max(v.norm(dim=-1).max() for v in f),20)
        self.assertGreater(f[3].norm(dim=-1).max(),0)
        untouched=r.elapsed[1].clone()
        r.reset(torch.tensor([0,3]))
        self.assertEqual(r.force[[0,3]].count_nonzero(),0)
        self.assertEqual(r.elapsed[[0,3]].count_nonzero(),0)
        self.assertEqual(r.elapsed[1],untouched)

    def test_zero_cycles_and_random_long_run(self):
        r=FootForceRamp(64,'cpu',zero_prob=1.)
        for _ in range(600): self.assertEqual(r.advance().count_nonzero(),0)
        r=FootForceRamp(64,'cpu')
        maximum=0.
        for _ in range(1200):
            f=r.advance()
            self.assertTrue(torch.isfinite(f).all())
            maximum=max(maximum,float(f.norm(dim=-1).max()))
            self.assertLessEqual(maximum,20.00001)
        self.assertGreater(maximum,15.)


if __name__=='__main__': unittest.main()
