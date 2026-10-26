"""Frozen original CEER finetune actor with explicit joint-reference input.

The learned privileged-state estimator is retained. The joint predictor is
never instantiated: its normalized output is replaced with our own reference.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from active_adaptation.learning.ppo.common import make_mlp

ROOT = Path(__file__).resolve().parents[2]
# Original checkpoint's AMASS metadata order, distinct from Isaac articulation order.
TARGET_JOINT_NAMES = [f'{side}_{joint}_joint' for side in ('left','right')
    for joint in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
TARGET_JOINT_NAMES += ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
TARGET_JOINT_NAMES += [f'{side}_{joint}_joint' for side in ('left','right')
    for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]

class FrozenCEER(nn.Module):
    def __init__(self, checkpoint=None, device='cpu'):
        super().__init__()
        path = Path(checkpoint) if checkpoint else ROOT / '.cache/ceer/checkpoint_final.pt'
        if not path.is_absolute():
            path = ROOT / path
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        assert ckpt['policy']['last_phase'] == 'finetune'
        cfg = ckpt['cfg']
        assert list(cfg.task.observation.policy) == ['boot_indicator_state','command','root_and_wrist_6d',
            'root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions']
        assert list(cfg.task.observation.policy.joint_pos_history.history_steps) == [0,1,2,3,4]
        self.action_cfg = cfg.task.action.copy()
        self.adapt = nn.Sequential(make_mlp([512,256]), nn.LazyLinear(256))
        self.actor = make_mlp([512,512,256])
        self.mean = nn.Linear(256,29)
        self.adapt(torch.zeros(1,257)); self.actor(torch.zeros(1,658))
        self.adapt.load_state_dict({k.removeprefix('module.'):v for k,v in ckpt['policy']['adapt_module'].items()}, strict=True)
        state = ckpt['policy']['actor_student']
        self.actor.load_state_dict({k.removeprefix('module.0.module.1.module.'):v for k,v in state.items()
                                   if k.startswith('module.0.module.1.module.')}, strict=True)
        self.mean.load_state_dict({k.removeprefix('module.0.module.2.module.actor_mean.'):v for k,v in state.items()
                                  if k.startswith('module.0.module.2.module.actor_mean.')}, strict=True)
        # The source checkpoint was finetuned with fixed ObservationNorm.
        # Use those exact transforms, not newly updated residual VecNorm stats.
        for name, index, dim in [('policy',2,257), ('joint',3,145)]:
            prefix = f'transforms.{index}'
            assert ckpt['env'][prefix+'.standard_normal'].item()
            for field in ('loc','scale'):
                value = ckpt['env'][prefix+'.'+field]
                assert value.shape == (dim,) and torch.isfinite(value).all()
                if field == 'scale': assert (value>0).all()
                self.register_buffer(name+'_'+field, value.clone())
        self.to(device).requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, proprioception_and_interface, joint_reference):
        assert proprioception_and_interface.shape[-1] == 257
        # Caller supplies five self-generated frames, in original dataset order.
        joint_reference = joint_reference.flatten(1)
        assert joint_reference.shape[-1] == 145
        obs = (proprioception_and_interface-self.policy_loc)/self.policy_scale
        joint = (joint_reference-self.joint_loc)/self.joint_scale
        latent = self.adapt(obs)
        return self.mean(self.actor(torch.cat((obs,latent,joint),-1)))
