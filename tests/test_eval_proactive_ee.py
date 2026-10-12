"""Metrics must not label avoiding contact or partial holds as success."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('ee_eval',Path(__file__).resolve().parents[1]/'scripts/eval_proactive_ee.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class EvaluationMetrics(unittest.TestCase):
    def rows(self):
        return [dict(time_s=i*.005,error_m=.01,normal_force_N=0.,positive_force_rate_N_s=0.,vx_m_s=.2,root_drift_m=0.) for i in range(1000)]
    def test_no_contact_is_not_a_precontact_measurement(self):
        result=module.summarize(self.rows(),True,'',3.)
        self.assertIsNone(result['contact_time_s'])
        self.assertIsNone(result['precontact_mean_vx_m_s'])
        self.assertTrue(result['tracking_success'])
    def test_partial_hold_is_not_success(self):
        result=module.summarize(self.rows(),False,'tilt',3.)
        self.assertFalse(result['tracking_success'])
    def test_precontact_window_excludes_contact(self):
        rows=self.rows()
        for r in rows:
            if r['time_s']>=1.:r['normal_force_N']=10.;r['vx_m_s']=0.
        result=module.summarize(rows,True,'',3.)
        self.assertEqual(result['contact_time_s'],1.)
        self.assertAlmostEqual(result['precontact_mean_vx_m_s'],.2)

class GradualTrajectory(unittest.TestCase):
    def test_waypoints_and_monotonic_insertion(self):
        import numpy as np
        a=np.array([.33,.20,.75]);g=np.array([.442,.20,.91]);wall=.43
        np.testing.assert_allclose(module.gradual_target(0,a,g,wall),a)
        np.testing.assert_allclose(module.gradual_target(2,a,g,wall),[.37,.20,.91])
        np.testing.assert_allclose(module.gradual_target(8,a,g,wall),g)
        np.testing.assert_allclose(module.gradual_target(12,a,g,wall),g)
        xs=[module.gradual_target(t,a,g,wall)[0] for t in np.linspace(0,12,601)]
        self.assertTrue(np.all(np.diff(xs)>=-1e-12))
        for t in [0,2,8]:
            velocity=(module.gradual_target(t+.0001,a,g,wall)-module.gradual_target(t-.0001,a,g,wall))/.0002
            self.assertLess(np.linalg.norm(velocity),1e-5)

if __name__=='__main__':unittest.main()
