"""Metric validity checks independent of Isaac and trained policies."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.compliance_eval.protocol import cases,envelope
from scripts.compliance_eval.report import behavior

class ComplianceProtocolTest(unittest.TestCase):
    def test_force_couples_and_budget(self):
        self.assertEqual(len(cases()),23)
        for c in cases():
            v=np.array([v for _,v in c['loads']],float)*20
            self.assertLessEqual(np.linalg.norm(v,axis=1).sum(),60)
            if c['group'] in ('yaw','bend'):np.testing.assert_allclose(v.sum(0),0)
        t=np.arange(0,12,.005);f=np.array([envelope(x) for x in t])*20
        self.assertLess(abs(np.diff(f)).max()/.005,32)
        self.assertEqual(envelope(2),0);self.assertEqual(envelope(8),0)

    def test_failure_not_reported_as_success_and_preforce_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'behavior';z=root/'behavior_zero';p.mkdir();z.mkdir()
            row=dict(name='root_forward',group='locomotion',direction=[1,0,0],loads=[['pelvis',[1,0,0]]],force_N=10,repeat=0)
            r=dict(rows=[row],dt=.02,seed=1,checkpoint_sha256='x',manifest_sha256='y',force_input_bodies=['pelvis'],joint_names=['left_knee_joint','right_knee_joint'])
            for path in (p,z):(path/'result.json').write_text(json.dumps(r))
            d=dict(root=np.zeros((601,1,3)),yaw=np.zeros((601,1)),torso_relative_quat=np.tile([1.,0,0,0],(601,1,1)),hands_b=np.zeros((601,1,2,3)),joint_pos=np.zeros((601,1,2)),root_velocity=np.zeros((601,1,3)),first_fall_step=np.array([-1]),initial_joint_pos=np.zeros((1,2)),initial_root=np.zeros((1,3)))
            np.savez(z/'traces.npz',**d)
            d['root'][101:,0,0]=.2;d['first_fall_step'][0]=200;np.savez(p/'traces.npz',**d)
            result=behavior(p)[0]
            self.assertFalse(result['directed_response_success']);self.assertIsNone(result['response_at_release'])
            d['root'][50,0,0]=.1;np.savez(p/'traces.npz',**d)
            with self.assertRaises(AssertionError):behavior(p)
if __name__=='__main__':unittest.main()
