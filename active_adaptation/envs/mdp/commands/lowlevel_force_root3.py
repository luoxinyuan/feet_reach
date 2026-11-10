"""Three-point physical-force command for full-joint-target CEER fine-tuning."""
from .online_estimated_compliance import OnlineEstimatedComplianceCommand
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply_inverse, quat_apply
from active_adaptation.utils.simple_locomotion_guidance import yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform


class LowLevelForceRoot3Command(OnlineEstimatedComplianceCommand):
    """Use the existing root3 force allocator without an EE/root action interface.

    G1_base's MotionTrackingCommand_impedance config contains several force
    budget knobs that belong only to that different command class. This task
    inherits the tracking dataset and joint selection, but the Jacobian command
    owns its force envelope, so discard those incompatible impedance-only keys.
    """
    def __init__(self, env, *args, **kwargs):
        for key in ('max_force', 'net_force_limit', 'net_torque_limit',
                    'force_safe_bounds', 'force_safe_default', 'force_penalty_offset'):
            kwargs.pop(key, None)
        super().__init__(env, *args, **kwargs)

    @observation
    def command(self):
        """Keep the pretrained low-level's 6D root locomotion command."""
        current_yaw = yaw_quat(self.asset.data.root_quat_w)
        target_yaw = yaw_quat(self._motion.root_quat_w[:, 0])
        vel_b = quat_apply_inverse(current_yaw, self._motion.root_lin_vel_w[:, 0])
        forward = torch.zeros(self.num_envs, 3, device=self.device)
        forward[:, 0] = 1.0
        heading_w = quat_apply(target_yaw, forward)
        heading_b = quat_apply_inverse(current_yaw, heading_w)
        root_height = self._motion.root_pos_w[:, 0, 2:3]
        force_limit = torch.full((self.num_envs, 1), 30.0, device=self.device)
        return torch.cat((root_height, vel_b[:, :2], heading_b[:, :2], force_limit), dim=-1)

    @observation
    def force_priv(self):
        """Keep the pretrained CEER privileged-observation width (6 force slots).

        The crouch-capable checkpoint was trained with 6 force slots. This
        experiment physically uses root/left-hand/right-hand; those three fill
        the first slots and the remaining slots stay zero, preserving its
        checkpoint/VecNorm tensor shape during fine-tuning.
        """
        n = self.num_envs
        pos = torch.zeros(n, 6, 3, device=self.device)
        root = self.asset.data.root_pos_w
        quat = self.asset.data.root_quat_w
        pos[:, :3] = quat_apply_inverse(
            quat[:, None].expand(-1, 3, -1),
            self.asset.data.body_pos_w[:, self.point_ids] - root[:, None])
        applied = torch.zeros_like(pos)
        applied[:, :3] = self.dense_force_gt().reshape(n, 3, 3) * 30.0
        expected = applied.clone()
        timer = self.remaining[:, None].clamp_min(0.0)
        return torch.cat((pos.flatten(1), applied.flatten(1), expected.flatten(1), timer), dim=-1)

    def force_priv_sym(self):
        # Three vector blocks, six slots each; mirror left/right and y sign.
        perm = []
        signs = []
        slot_perm = [0, 2, 1, 3, 4, 5]
        for _ in range(3):
            perm.extend([3 * slot_perm[i] + c for i in range(6) for c in range(3)])
            signs.extend(([1., -1., 1.] * 6))
        perm.append(54); signs.append(1.)
        return SymmetryTransform(perm, signs)
