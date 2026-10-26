import unittest
import numpy as np
from scripts.analyze_gentle_continuous_suite import extra_metrics

class MetricsTests(unittest.TestCase):
    def samples(self):
        p=np.zeros((200,9));p[:,0]=np.arange(1,201)*.005;p[:,5]=.45
        return p

    def test_sliding_window_catches_late_impulse(self):
        p=self.samples();p[:20,1]=1.;p[100:120,1]=100.
        m=extra_metrics(p,.2,.4,True)
        self.assertAlmostEqual(m['max_sliding_impulse_100ms_Ns'],10.)

    def test_contact_success_independent_of_precision_but_requires_stability(self):
        p=self.samples();p[:,1]=20.;p[:,3]=10.
        self.assertTrue(extra_metrics(p,.2,.4,True)['contact_hold_success'])
        self.assertFalse(extra_metrics(p,.2,.4,False)['contact_hold_success'])
        self.assertEqual(extra_metrics(p,.2,.4,True)['hold_contact_fraction'],1.)

    def test_detachments_exclude_initial_approach_and_short_gaps(self):
        p=self.samples();p[50:,1]=20.;p[100:104,1]=0;p[150:153,1]=0
        self.assertEqual(extra_metrics(p,.2,.4,True)['detachments_20ms'],1)

if __name__=='__main__':unittest.main()
