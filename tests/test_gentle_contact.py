import unittest
import torch
from active_adaptation.utils.gentle_contact import ContactEvents


class ContactEventTests(unittest.TestCase):
    def make(self):
        return ContactEvents(1, 'cpu', .005), torch.tensor([True])

    def test_constant_force_cost_and_window(self):
        event, present = self.make()
        cost = 0.
        for _ in range(20):
            dj, _, _ = event.advance(torch.tensor([31.]), present)
            cost += dj.item()
        self.assertAlmostEqual(event.impulse.item(), 3.1, places=5)
        self.assertAlmostEqual(cost, 6., places=4)
        dj, dp, growth = event.advance(torch.tensor([1000.]), present)
        self.assertEqual((dj+dp+growth).item(), 0.)

    def test_onset_charged_before_early_termination(self):
        event, present = self.make()
        dj, dp, growth = event.advance(torch.tensor([1000.]), present)
        self.assertEqual(dj.item(), 10.)
        self.assertEqual(dp.item(), 5.)
        self.assertEqual(growth.item(), .5)
        event.reset(torch.tensor([0]))
        self.assertEqual(event.impulse.item(), 0.)
        self.assertTrue(event.armed.item())

    def test_repeated_contact_and_chatter(self):
        event, present = self.make()
        for _ in range(20):event.advance(torch.tensor([31.]), present)
        for _ in range(5):event.advance(torch.tensor([0.]), present)
        event.advance(torch.tensor([31.]), present)
        self.assertEqual(event.count.item(), 1)
        for _ in range(20):event.advance(torch.tensor([0.]), present)
        cost=0.
        for _ in range(20):cost+=event.advance(torch.tensor([31.]), present)[0].item()
        self.assertEqual(event.count.item(), 2)
        self.assertAlmostEqual(cost, 6., places=4)

    def test_no_wall_and_free_impulse(self):
        event, _ = self.make()
        dj, dp, growth = event.advance(torch.tensor([1000.]), torch.tensor([False]))
        self.assertEqual((dj+dp+growth).item(), 0.)
        event.reset(torch.tensor([0]))
        cost=0.
        for _ in range(20):cost+=event.advance(torch.tensor([19.]), torch.tensor([True]))[0].item()
        self.assertEqual(cost, 0.)

if __name__ == '__main__':unittest.main()
