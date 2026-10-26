"""Validate portable residual/reference adapters against the native frozen models."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import numpy as np
import torch
from mujoco_whole_body_student import make_simulation
from active_adaptation.utils.sonic import FrozenSonic,ResidualReference,ProprioceptionHistory
from active_adaptation.utils.ceer_residual import FrozenCEER

torch.set_num_threads(1);torch.manual_seed(42)
report={}
for backend in ('sonic','ceer'):
 sim=make_simulation(policy=backend)
 low=torch.tensor(sim.meta['joint_lower'])[None];high=torch.tensor(sim.meta['joint_upper'])[None]
 ref=ResidualReference(1,low,high)
 ref.reset(torch.tensor([0]),torch.from_numpy(sim.data.qpos[sim.qadr]).float()[None])
 reference_errors=[]
 for _ in range(100):
  raw=torch.randn(1,29);actual=torch.from_numpy(sim.data.qpos[sim.qadr]).float()[None]
  ref.update(raw,actual);sim.integrate_reference(raw[0].numpy())
  reference_errors.append(float(np.max(np.abs(sim.reference-ref.q[0].numpy()))))
 assert max(reference_errors)<2e-6,reference_errors
 before=sim.reference.copy();sim.integrate_reference(np.zeros(29));assert np.array_equal(before,sim.reference)
 sim.reset();original_obs=sim.observation().copy();sim.step(np.zeros((9,3)));sim.reset()
 np.testing.assert_array_equal(sim.observation(),original_obs)
 assert sim.actions.max()==0 and sim.low_actions.max()==0 and sim.reference_velocity.max()==0
 errors=[]
 native=FrozenSonic(device='cpu') if backend=='sonic' else FrozenCEER(device='cpu')
 for step in range(12):
  force=np.zeros((9,3));force[4]=[5.,1.,0.]
  sim.step(force)
  if step%sim.decimation:continue
  if backend=='sonic':
   encoder,prop=sim.last_low_obs
   q=torch.from_numpy(encoder[:,4:294].reshape(1,10,29).copy())
   orientation=torch.from_numpy(encoder[:,601:661].reshape(1,10,6).copy())
   expected=native(q,orientation,torch.from_numpy(prop))[0].numpy()
  else:
   obs,target=sim.last_low_obs;expected=native(torch.from_numpy(obs),torch.from_numpy(target))[0].numpy()
  errors.append(float(np.max(np.abs(expected-sim.low_action))))
 assert max(errors)<1e-4,errors
 assert sim.last_obs.shape==(381,)
 report[backend]=dict(reference_integrator_error=max(reference_errors),native_low_level_error=max(errors),reset_verified=True,public_observation_dim=381)
 del native
out=ROOT/'artifacts/mujoco_residual';out.mkdir(parents=True,exist_ok=True)
(out/'adapter_parity.json').write_text(json.dumps(report,indent=2));print(report)
