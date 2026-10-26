"""SONIC action adapter. Ground-truth motion is deliberately inaccessible here."""
import torch
from isaaclab.utils.math import matrix_from_quat, quat_mul, quat_conjugate
from active_adaptation.utils.math import yaw_quat
from active_adaptation.utils.sonic import (
    FrozenSonic, ResidualReference, ProprioceptionHistory, JOINT_NAMES, motor_parameters,
)
from .action import ActionManager
from active_adaptation.utils.symmetry import joint_space_symmetry

class SonicResidualAction(ActionManager):
    def __init__(self, env, reference_speed=2., max_reference_error=.6, **kwargs):
        super().__init__(env)
        if abs(env.step_dt - .02) > 1e-6:
            raise ValueError('SONIC requires 50 Hz control (step_dt=0.02).')
        if env.cfg.robot.name != 'g1_sonic':
            raise ValueError('Use g1_sonic asset to match SONIC gains and action scales.')
        self.joint_names = JOINT_NAMES
        self.joint_ids = [self.asset.joint_names.index(n) for n in JOINT_NAMES]
        self.action_dim = 29
        self.sonic = FrozenSonic(device=self.device)
        params = motor_parameters().to(self.device)
        self.nominal, self.action_scaling = params[0], params[-1]
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        limits = self.asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.reference = ResidualReference(self.num_envs, limits[..., 0], limits[..., 1],
                                           env.step_dt, reference_speed, max_reference_error)
        self.history = ProprioceptionHistory(self.num_envs, self.device)
        self.action_buf = torch.zeros(self.num_envs, 3, 29, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, 29, device=self.device)
        self.low_action = torch.zeros_like(self.applied_action)
        self.target = self.default_joint_pos.clone()

    def symmetry_transforms(self):
        return joint_space_symmetry(self.asset, self.joint_names)

    @torch.no_grad()
    def reset(self, env_ids):
        self.reference.reset(env_ids, self.asset.data.joint_pos[:, self.joint_ids])
        self.history.reset(env_ids)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0
        self.low_action[env_ids] = 0
        self.target[env_ids] = self.asset.data.joint_pos[env_ids]

    def orientation(self):
        # Upright at measured heading: no AMASS orientation, no fixed world-yaw
        # restoring target. Turning must emerge from forces and joint residuals.
        q = self.asset.data.root_quat_w
        relative = quat_mul(quat_conjugate(q), yaw_quat(q))
        return matrix_from_quat(relative)[..., :2].reshape(self.num_envs, 6)

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            raw = tensordict['action']
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = raw
            self.applied_action.copy_(raw.tanh())
            data = self.asset.data
            q = data.joint_pos[:, self.joint_ids]
            self.reference.update(raw, q)
            obs = self.history.push(data.root_ang_vel_b, q - self.nominal,
                                    data.joint_vel[:, self.joint_ids], self.low_action,
                                    data.projected_gravity_b)
            self.low_action.copy_(self.sonic(self.reference.q, self.orientation(), obs))
            self.target[:, self.joint_ids] = self.nominal + self.action_scaling * self.low_action
        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()
