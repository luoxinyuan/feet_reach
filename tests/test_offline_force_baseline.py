import unittest,json
from pathlib import Path
import numpy as np
from scripts.offline_force_baseline.solver import solve_sequence
from active_adaptation.utils.segmented_guidance import geometry_jacobians
from scripts.compliance_eval.protocol import load_arrays

class OfflineForceBaselineTest(unittest.TestCase):
 def test_opposing_forces_cannot_evade_budget(self):
  A=np.zeros((4,2,6));A[:,0,0]=1;A[:,1,3]=1
  dq=np.tile([.1,-.1],(4,1));f,r=solve_sequence(A,dq,10,slew=150)
  self.assertLessEqual(np.linalg.norm(f,axis=-1).sum(-1).max(),10.0001)
  self.assertLess(np.linalg.norm(f.sum(1),axis=-1).max(),1e-5)
  self.assertGreater(r['best_rmse_rad'],.03)
 def test_temporal_bound_is_inside_solver(self):
  A=np.zeros((6,1,3));A[:,0,0]=1;dq=np.array([[.2]]*3+[[-.2]]*3)
  f,_=solve_sequence(A,dq,30,slew=150)
  self.assertLessEqual(np.linalg.norm(np.diff(f,axis=0),axis=-1).max(),3.00001)
 def test_block_boundary_slew(self):
  A=np.zeros((1,1,3));A[:,0,0]=1;dq=np.array([[-.2]])
  previous=np.array([[8.,0,0]])
  f,_=solve_sequence(A,dq,30,slew=150,previous_force=previous)
  self.assertLessEqual(np.linalg.norm(f[0]-previous,axis=-1).max(),3.00001)
 def test_offset_point_jacobian_finite_difference(self):
  root=Path(__file__).resolve().parents[1];m,a=load_arrays(root/'dataset/passive_whole_body_full/val')
  q=np.asarray(a['joint_pos'][0:1],float);rp=np.asarray(a['root_pos_w'][0:1],float);rq=np.asarray(a['root_quat_w'][0:1],float)
  g=json.loads((root/'artifacts/ceer_segmented_jacobian/geometry.json').read_text());points=[dict(body='left_hand_mimic',offset=[.05,0,0])]
  _,j=geometry_jacobians(q,rp,rq,m['joint_names'],points,g)
  for name in ['left_wrist_yaw_joint','waist_pitch_joint','left_knee_joint']:
   idx=m['joint_names'].index(name);plus=q.copy();minus=q.copy();plus[0,idx]+=1e-6;minus[0,idx]-=1e-6
   p1,_=geometry_jacobians(plus,rp,rq,m['joint_names'],points,g);p0,_=geometry_jacobians(minus,rp,rq,m['joint_names'],points,g)
   np.testing.assert_allclose((p1-p0)[0,0]/2e-6,j[0,0,:,6+idx],atol=1e-7)
if __name__=='__main__':unittest.main()
