"""Nominal proactive reaching targets for a frozen CEER controller."""
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.proactive_ee import ProactiveEECommand
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform


class ProactiveResidualCommand(ProactiveEECommand):
    def __init__(self, env, *args, ceer_force_limit=30., **kwargs):
        super().__init__(env, *args, **kwargs)
        self.ceer_force_limit = ceer_force_limit

    @observation
    def command(self):
        # Same six fields and frame convention as CEER's impedance command.
        q = self.asset.data.root_quat_w
        velocity = quat_apply_inverse(q, self._motion.root_lin_vel_w[:, 0])[:, :2]
        heading = torch.zeros(self.num_envs, 3, device=self.device)
        heading[:, 0] = 1.
        heading = quat_apply(yaw_quat(self._motion.root_quat_w[:, 0]), heading)
        heading = quat_apply_inverse(yaw_quat(q), heading)[:, :2]
        limit = torch.full((self.num_envs, 1), self.ceer_force_limit, device=self.device)
        return torch.cat([self._motion.root_pos_w[:, 0, 2:3], velocity, heading, limit], -1)

    def command_sym(self):
        return SymmetryTransform(torch.arange(6), [1., 1., -1., 1., -1., 1.])

    def get_root_and_wrist_6d_reference(self):
        reference = super().get_root_and_wrist_6d_reference().clone()
        # The task goal is expressed relative to the live root. Keep the right
        # hand and both orientations from the nominal motion. Never feed the
        # residual back into the task goal or the reward.
        reference[:, :3] = self.target_ee_pos_b()
        return reference
