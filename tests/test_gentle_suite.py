import unittest
import numpy as np
from scripts.eval_gentle_suite import summarize


class MetricsTests(unittest.TestCase):
    def samples(self):
        p=np.zeros((100,9));p[:,0]=np.arange(1,101)*.005
        p[:,1]=20.;p[:,5]=.45
        c=np.zeros((25,15));c[:,0]=np.arange(1,26)*.02
        return p,c

    def test_success_and_impulse(self):
        p,c=self.samples();r=summarize(p,c,0.,.4,'')
        self.assertTrue(r['success']);self.assertEqual(r['impulse_100ms_Ns'],2.)
        self.assertEqual(r['max_rate_N_s'],4000.)

    def test_contact_is_not_success_after_fall_or_tangent_error(self):
        p,c=self.samples();self.assertFalse(summarize(p,c,0.,.4,'feet_drift')['success'])
        p[:,3]=.04;self.assertFalse(summarize(p,c,0.,.4,'')['success'])

    def test_brief_or_absent_contact_not_success(self):
        p,c=self.samples();p[::50,1]=0
        self.assertFalse(summarize(p,c,0.,.4,'')['success'])
        p[:,1]=0;r=summarize(p,c,0.,.4,'')
        self.assertFalse(r['success']);self.assertIsNone(r['impulse_100ms_Ns'])
