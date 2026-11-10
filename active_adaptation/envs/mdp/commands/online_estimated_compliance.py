"""Online Jacobian prompts with causal CEER-style force adaptation."""
import torch
from .jacobian_compliance import JacobianComplianceCommand
from .dense_link_compliance import DenseLinkComplianceCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply_inverse, quat_apply, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry


class OnlineEstimatedComplianceCommand(JacobianComplianceCommand):
    # Reuse the tested GT labels and causal proprioception history only;
    # no offline force-cache constructor or force application is inherited.
    dense_force_gt = DenseLinkComplianceCommand.dense_force_gt
    dense_force_gt_sym = DenseLinkComplianceCommand.dense_force_gt_sym
    dense_estimator_history = DenseLinkComplianceCommand.dense_estimator_history
    dense_estimator_history_sym = DenseLinkComplianceCommand.dense_estimator_history_sym

    def __init__(self, env, *args, estimator_history=10, expected_force_points=3,
                 force_prompt_history_offsets=(0, 1, 2, 4, 8, 16), **kwargs):
        for key in ('max_force', 'net_force_limit', 'net_torque_limit',
                    'force_safe_bounds', 'force_safe_default', 'force_penalty_offset'):
            kwargs.pop(key, None)
        super().__init__(env, *args, **kwargs)
        self.point_names = [self.asset.body_names[int(i)] for i in self.point_ids]
        if len(self.point_names) != int(expected_force_points):
            raise ValueError(f'Expected {expected_force_points} force points, got {len(self.point_names)}.')
        self.estimator_history_steps = int(estimator_history)
        if self.estimator_history_steps < 1:
            raise ValueError('History must contain at least one frame.')
        self.force_history = torch.zeros(self.num_envs, self.estimator_history_steps, 93, device=self.device)
        self.history_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_tick = -1
        self.force_prompt_history_offsets = tuple(int(x) for x in force_prompt_history_offsets)
        if not self.force_prompt_history_offsets or self.force_prompt_history_offsets[0] != 0:
            raise ValueError('Force prompt history offsets must start with the current frame (0).')
        if tuple(sorted(set(self.force_prompt_history_offsets))) != self.force_prompt_history_offsets:
            raise ValueError('Force prompt history offsets must be unique and sorted oldest-last.')
        self.force_prompt_dim = 3 * len(self.point_names)
        self.force_prompt_history = torch.zeros(
            self.num_envs, max(self.force_prompt_history_offsets) + 1,
            self.force_prompt_dim, device=self.device)
        self.force_prompt_history_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_prompt_history_tick = -1

    @property
    def point_force_buffer(self):
        # Canonical point order, independent of randomized active slot order.
        return self.force_apply_buffer[:, self.point_ids]

    def reset(self, ids):
        super().reset(ids)
        self.force_history[ids] = 0.
        self.history_fresh[ids] = True
        self.force_prompt_history[ids] = 0.
        self.force_prompt_history_fresh[ids] = True

    @observation
    def dense_force_gt_history(self):
        """Current per-point forces and sampled past frames, zero-padded at reset."""
        current = self.dense_force_gt()
        tick = int(self.log_step)
        if tick != self.force_prompt_history_tick:
            self.force_prompt_history[:, 1:] = self.force_prompt_history[:, :-1].clone()
            self.force_prompt_history[:, 0] = current
            self.force_prompt_history_tick = tick
        fresh = self.force_prompt_history_fresh
        self.force_prompt_history[fresh] = 0.
        self.force_prompt_history[fresh, 0] = current[fresh]
        self.force_prompt_history_fresh.zero_()
        frames = self.force_prompt_history[:, self.force_prompt_history_offsets]
        return frames.reshape(self.num_envs, -1)

    def dense_force_gt_history_sym(self):
        dim = len(self.force_prompt_history_offsets) * self.force_prompt_dim
        return SymmetryTransform(torch.arange(dim), torch.ones(dim))

    @observation
    def command(self):
        current_yaw = yaw_quat(self.asset.data.root_quat_w)
        target_yaw = yaw_quat(self._motion.root_quat_w[:, 0])
        vel_b = quat_apply_inverse(current_yaw, self._motion.root_lin_vel_w[:, 0])
        forward = torch.zeros(self.num_envs, 3, device=self.device)
        forward[:, 0] = 1.0
        heading_b = quat_apply_inverse(current_yaw, quat_apply(target_yaw, forward))
        root_height = self._motion.root_pos_w[:, 0, 2:3]
        force_limit = torch.full((self.num_envs, 1), 30.0, device=self.device)
        return torch.cat((root_height, vel_b[:, :2], heading_b[:, :2], force_limit), dim=-1)

    def command_sym(self):
        return SymmetryTransform(torch.arange(6), [1., 1., -1., 1., -1., 1.])

    @observation
    def target_joint_pos_obs(self):
        # CEER low-level training uses the complete 29-joint trajectory (five
        # reference frames), not the sparse 17-joint foot-reach target helper.
        return self._motion.joint_pos.reshape(self.num_envs, -1)

    def target_joint_pos_obs_sym(self):
        return joint_space_symmetry(self.asset, self.dataset.joint_names).repeat(len(self.future_steps))

    @observation
    def force_priv(self):
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
        perm = []; signs = []
        slot_perm = [0, 2, 1, 3, 4, 5]
        for _ in range(3):
            perm.extend([3 * slot_perm[i] + c for i in range(6) for c in range(3)])
            signs.extend([1., -1., 1.] * 6)
        perm.append(54); signs.append(1.)
        return SymmetryTransform(perm, signs)
