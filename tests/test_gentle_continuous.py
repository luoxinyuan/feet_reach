import unittest
import torch
from active_adaptation.utils.gentle_contact import ContinuousForceRates

class ContinuousTests(unittest.TestCase):
    def test_late_peak_and_no_cap(self):
        e=ContinuousForceRates(1,'cpu',.005)
        yes=torch.tensor([True])
        for _ in range(100):e.advance(torch.tensor([10.]),yes,yes)
        rise,fall=e.advance(torch.tensor([210.]),yes,yes)
        self.assertAlmostEqual(rise.item(),7.9,places=5)
        self.assertEqual(fall.item(),0.)
        rise,fall=e.advance(torch.tensor([210.]),yes,yes)
        self.assertEqual((rise+fall).item(),0.)

    def test_unloading_and_task_withdrawal(self):
        yes=torch.tensor([True]);no=~yes
        for required,expected in [(yes,3.9),(no,0.)]:
            e=ContinuousForceRates(1,'cpu',.005)
            e.advance(torch.tensor([100.]),yes,yes)
            rise,fall=e.advance(torch.tensor([0.]),yes,required)
            self.assertAlmostEqual(fall.item(),expected,places=5)
            self.assertEqual(rise.item(),0.)

    def test_deadband_wall_and_reset(self):
        e=ContinuousForceRates(1,'cpu',.005);yes=torch.tensor([True])
        rise,fall=e.advance(torch.tensor([2.5]),yes,yes)
        self.assertEqual((rise+fall).item(),0.)
        rise,fall=e.advance(torch.tensor([100.]),~yes,yes)
        self.assertEqual((rise+fall).item(),0.)
        e.reset(torch.tensor([0]));self.assertEqual(e.previous.item(),0.)
        self.assertFalse(e.established.item())
