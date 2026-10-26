"""Compare the exported student with the native training PPO rollout on CPU."""
from pathlib import Path
import os,sys,json
os.environ['ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from types import SimpleNamespace
from active_adaptation.learning.ppo.ppo import PPOPolicy
from active_adaptation.utils.symmetry import SymmetryTransform
from torchrl.data import CompositeSpec,UnboundedContinuousTensorSpec
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type,ExplorationType
root=str(Path(__file__).resolve().parents[2]/'artifacts/mujoco_whole_body_student')+'/'
meta=json.load(open(root+'metadata.json'))
c=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
torch.set_num_threads(1)
dims={'policy':265,'priv':323,'joint_target':29,'priv_critic':3}
class Obs:
 def __init__(self,n):self.n=n
 def symmetry_transforms(self):return SymmetryTransform(torch.arange(self.n),torch.ones(self.n))
env=SimpleNamespace(action_manager=SimpleNamespace(joint_names=meta['joint_names'],symmetry_transforms=Obs(29).symmetry_transforms),cfg=SimpleNamespace(reward={}),observation_funcs={k:Obs(n) for k,n in dims.items()})
spec=CompositeSpec({k:UnboundedContinuousTensorSpec((2,n)) for k,n in dims.items()},shape=(2,))
p=PPOPolicy(c['cfg']['algo'],spec,UnboundedContinuousTensorSpec((2,29)),UnboundedContinuousTensorSpec((2,1)),device='cpu',env=env)
p.load_state_dict(c['policy']);p.eval();roll=p.get_rollout_policy('deploy')
exported=torch.jit.load(root+'student.pt')
errors=[]
with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
 for _ in range(10):
  x=torch.randn(2,265)
  native=roll(TensorDict({'policy':x},batch_size=[2]))['action']
  errors.append(float((native-exported(x)).abs().max()))
print('MAX ERROR',max(errors),'INPUTS',roll.in_keys)
assert max(errors)<1e-6
json.dump({'native_student_export_max_error':max(errors),'rollout_input_keys':list(roll.in_keys)},open(root+'native_export_test.json','w'),indent=2)
