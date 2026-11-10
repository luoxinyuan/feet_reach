"""Force-free full-motion tracking command for the CEER low-level policy."""

import torch

from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from active_adaptation.utils import symmetry as sym_utils


class CEERMotionTrackingCommand(MotionTrackingCommand):
    """Motion tracking with joint targets and root-motion cues, without force I/O."""

    supports_symmetry_augmentation = False

    def before_update(self):
        # MotionTrackingCommand's impedance/compliance subclasses populate these
        # reward targets in their override; this force-free command still uses
        # the same root tracking rewards, so initialize the targets directly.
        super().before_update()
        self.update_reward_target()

    @observation
    def command(self):
        """Return root height, planar target velocity, and target heading (5D)."""
        root_height = self._motion.root_pos_w[:, 0, 2:3]

        current_quat = self.asset.data.root_quat_w
        target_linvel_w = self._motion.root_lin_vel_w[:, 0]
        target_linvel_b = quat_apply_inverse(current_quat, target_linvel_w)[:, :2]

        current_yaw = yaw_quat(current_quat)
        target_yaw = yaw_quat(self._motion.root_quat_w[:, 0])
        heading_w = quat_apply(
            target_yaw,
            torch.tensor([1.0, 0.0, 0.0], device=self.device)
            .expand(self.num_envs, -1),
        )
        heading_b = quat_apply_inverse(current_yaw, heading_w)[:, :2]

        return torch.cat([root_height, target_linvel_b, heading_b], dim=-1)

    def command_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]),
        ])
