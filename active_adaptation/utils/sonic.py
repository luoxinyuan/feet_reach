"""Frozen official SONIC G1 controller and causally generated joint references.

This module intentionally has no command manager, dataset or motion-target API.
Joint order is the official G1 Isaac Lab order, resolved by name in the adapter.
"""
from pathlib import Path
import sys
import hashlib
import json
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]

def motor_parameters():
    """Official robots/g1.py MODEL_12 gains, default pose, and action scaling."""
    values = []
    for name in JOINT_NAMES:
        if any(s in name for s in ('hip_pitch', 'hip_roll', 'knee')):
            arm, effort, vel = .025101925, 139., 20.
        elif 'hip_yaw' in name or 'waist_yaw' in name:
            arm, effort, vel = .010177520, 88., 32.
        elif 'ankle' in name or 'waist' in name:
            arm, effort, vel = 2 * .003609725, 50., 37.
        elif 'wrist_pitch' in name or 'wrist_yaw' in name:
            arm, effort, vel = .00425, 5., 22.
        else:
            arm, effort, vel = .003609725, 25., 37.
        q = 0.
        for key, value in [('hip_pitch', -.312), ('knee', .669), ('ankle_pitch', -.363),
                           ('elbow', .6), ('shoulder_pitch', .2), ('shoulder_roll', .2)]:
            if key in name:
                q = -value if key == 'shoulder_roll' and name.startswith('right') else value
        kp, kd = arm * (20 * torch.pi)**2, 4 * arm * (20 * torch.pi)
        values.append([q, kp, kd, arm, effort, vel, .25 * effort / kp])
    return torch.tensor(values, dtype=torch.float32).T

class FrozenSonic(nn.Module):
    """Native official G1 encoder + FSQ + dynamics decoder, arbitrary batch size.

    Only inactive modality/auxiliary modules are omitted. All retained parameters
    must match the official checkpoint exactly (strict load).
    """
    def __init__(self, cache=None, device='cpu'):
        super().__init__()
        cache = Path(cache) if cache else ROOT / '.cache/sonic'
        manifest = json.loads((cache / 'manifest.json').read_text())
        if manifest['variant'] != 'sonic_release':
            raise ValueError('This observation adapter is validated for sonic_release only.')
        for name in ('config.yaml', 'actor.pt'):
            digest = hashlib.sha256((cache / name).read_bytes()).hexdigest()
            if digest != manifest['files'][name]:
                raise ValueError(f'SONIC resource checksum mismatch: {name}')
        for folder in (cache / 'source', cache / 'deps'):
            sys.path.insert(0, str(folder))
        from omegaconf import OmegaConf
        from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
        cfg_text = (cache / 'config.yaml').read_text()
        cfg_text = cfg_text.replace('groot.rl.agents.modules.modules.BaseModule', 'gear_sonic.trl.modules.base_module.BaseModule')
        cfg = OmegaConf.create(cfg_text)
        assert not cfg.algo.config.actor.running_mean_std
        b = cfg.algo.config.actor.backbone
        assert b.num_future_frames == 10 and b.max_num_tokens == 2
        self.frames = 10
        dims = {'encoder_index': [3], 'command_multi_future_nonflat': [10, 58],
                'motion_anchor_ori_b_mf_nonflat': [10, 6]}
        env = OmegaConf.create({'obs': {'group_obs_dims': {'tokenizer': dims},
                                      'group_obs_names': {'tokenizer': list(dims)}},
                                'robot': {'actions_dim': 29, 'algo_obs_dim_dict': {'actor_obs': 930}}})
        self.model = UniversalTokenModule(
            env_config=env, algo_config=cfg.algo.config,
            proprioception_features=['actor_obs'], num_future_frames=10,
            num_fsq_levels=32, fsq_level_list=32, max_num_tokens=2,
            quantizer=b.quantizer, encoders=b.encoders, decoders=b.decoders,
            active_encoders=['g1'], active_decoders=['g1_dyn'],
            encoder_sample_probs={'g1': 1., 'teleop': 0., 'smpl': 0.},
            reencode_smpl_g1_recon=False,
        )
        state = torch.load(cache / 'actor.pt', map_location='cpu', weights_only=True)
        prefix = 'actor_module.'
        expected = self.model.state_dict()
        selected = {k: state[prefix + k] for k in expected}
        self.model.load_state_dict(selected, strict=True)
        self.to(device).requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Even a parent module's train() must never enable dropout/stat updates.
        return super().train(False)

    @torch.no_grad()
    def forward(self, q_reference, orientation_6d, proprioception, qd_reference=None):
        n = q_reference.shape[0]
        if q_reference.ndim == 2:
            q_reference = q_reference[:, None].expand(-1, self.frames, -1)
        if qd_reference is None:
            qd_reference = torch.zeros_like(q_reference)
        if orientation_6d.ndim == 2:
            orientation_6d = orientation_6d[:, None].expand(-1, self.frames, -1)
        assert q_reference.shape == qd_reference.shape == (n, 10, 29)
        assert orientation_6d.shape == (n, 10, 6) and proprioception.shape == (n, 930)
        indices = q_reference.new_zeros(n, 3)
        indices[:, 0] = 1.
        # Official command_multi_future concatenates all q frames, then all qd
        # frames, before reshaping to (10, 58); do not interleave q/qd.
        command = torch.cat((q_reference.flatten(1), qd_reference.flatten(1)), -1)
        tokenizer = torch.cat((indices, command.flatten(1), orientation_6d.flatten(1)), -1)
        return self.model({'actor_obs': proprioception[:, None], 'tokenizer': tokenizer[:, None]})[:, 0]

class ResidualReference:
    """Persistent joint reference; only reset(actual_q) and student actions write it."""
    def __init__(self, num_envs, lower, upper, dt=.02, speed=2., max_error=.6):
        self.lower, self.upper = lower, upper
        self.dt, self.speed, self.max_error = dt, speed, max_error
        self.q = torch.zeros(num_envs, 29, device=lower.device)
        self.velocity = torch.zeros_like(self.q)

    def reset(self, ids, actual_q):
        self.q[ids] = actual_q[ids].clamp(self.lower[ids], self.upper[ids])
        self.velocity[ids] = 0.

    def update(self, action, actual_q):
        # Clamp proposed increment, not the held reference: zero action holds
        # exactly, even if an external force displaces the real robot.
        lower = torch.maximum(self.lower, actual_q - self.max_error)
        upper = torch.minimum(self.upper, actual_q + self.max_error)
        proposal = (self.q + self.dt * self.speed * action.tanh()).clamp(lower, upper)
        delta = proposal - self.q
        requested = self.dt * self.speed * action.tanh()
        delta = torch.where(delta * requested > 0, delta.sign() * torch.minimum(delta.abs(), requested.abs()), 0.)
        self.q.add_(delta)
        self.velocity.copy_(delta / self.dt)

class ProprioceptionHistory:
    """SONIC term-major, oldest-first histories, with independent environment reset."""
    def __init__(self, n, device):
        self.values = [torch.zeros(n, 10, width, device=device) for width in (3, 29, 29, 29, 3)]
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.fresh[ids] = True
        for value in self.values:
            value[ids] = 0.

    def push(self, angular_velocity, relative_q, qd, last_action, gravity):
        for history, current in zip(self.values, (angular_velocity, relative_q, qd, last_action, gravity)):
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
            history[self.fresh] = current[self.fresh, None]
        self.fresh.zero_()
        return torch.cat([x.flatten(1) for x in self.values], -1)
