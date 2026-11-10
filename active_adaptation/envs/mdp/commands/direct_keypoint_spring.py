"""Independent next-frame anchor springs, without wrench allocation."""
import torch

from .expanded_spring_compliance import ExpandedSpringComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.body_guidance import limit_force


class DirectKeypointSpringCommand(ExpandedSpringComplianceCommand):
    def __init__(self, env, *args, spring_gain=80.,
                 estimator_history=10, expected_force_points=3,
                 force_prompt_history_offsets=(0, 1, 2, 4, 8, 16), **kwargs):
        # The filtered force-history base task inherits these Jacobian-only
        # settings; direct springs have no Jacobian allocator to consume them.
        for key in ('joint_delta_gain', 'joint_feedback_gain', 'force_penalty',
                    'smooth_penalty', 'resample_attempts'):
            kwargs.pop(key, None)
        super().__init__(env, *args, **kwargs)
        if len(self.point_names) != int(expected_force_points) or len(self.point_names) != 3:
            raise ValueError('Direct root/EE springs require exactly three points.')
        if spring_gain <= 0:
            raise ValueError('spring_gain must be positive.')
        self.spring_gain = float(spring_gain)
        self.force_prompt_history_offsets = tuple(int(x) for x in force_prompt_history_offsets)
        if (not self.force_prompt_history_offsets or self.force_prompt_history_offsets[0] != 0
                or tuple(sorted(set(self.force_prompt_history_offsets))) != self.force_prompt_history_offsets):
            raise ValueError('Force prompt history offsets must be unique, sorted, and start at zero.')
        self.force_prompt_history = torch.zeros(
            self.num_envs, max(self.force_prompt_history_offsets) + 1, 9, device=self.device)
        self.force_prompt_history_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_prompt_history_tick = -1
        self.estimator_history_steps = int(estimator_history)
        if self.estimator_history_steps < 1:
            raise ValueError('Estimator history must contain at least one frame.')
        self.force_history = torch.zeros(self.num_envs, self.estimator_history_steps, 93, device=self.device)
        self.history_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_tick = -1

    def reset(self, ids):
        super().reset(ids)
        self.force_prompt_history[ids] = 0.
        self.force_prompt_history_fresh[ids] = True
        self.force_history[ids] = 0.
        self.history_fresh[ids] = True

    @observation
    def dense_force_gt(self):
        """Actual applied root/hand spring forces, in root coordinates and /30 N."""
        q = self.asset.data.root_quat_w[:, None].expand(-1, len(self.point_names), -1)
        return quat_apply_inverse(q, self.point_force_buffer).flatten(1) / 30.

    def dense_force_gt_sym(self):
        dim = 3 * len(self.point_names)
        return SymmetryTransform(torch.arange(dim), torch.ones(dim))

    @observation
    def dense_force_gt_history(self):
        """Current spring forces and causal history, zero-padded after reset."""
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
        return self.force_prompt_history[:, self.force_prompt_history_offsets].reshape(self.num_envs, -1)

    def dense_force_gt_history_sym(self):
        dim = len(self.force_prompt_history_offsets) * 9
        return SymmetryTransform(torch.arange(dim), torch.ones(dim))

    @observation
    def dense_estimator_history(self):
        manager = self.env.action_manager
        data = self.asset.data
        current = torch.cat((data.root_ang_vel_b / 5., data.projected_gravity_b,
            data.joint_pos[:, manager.joint_ids], data.joint_vel[:, manager.joint_ids] / 10.,
            manager.low_action.clamp(-10, 10) / 5.), -1)
        if self.history_tick != self.log_step:
            self.force_history[:, 1:] = self.force_history[:, :-1].clone()
            self.force_history[:, 0] = current
            self.history_tick = self.log_step
        self.force_history[self.history_fresh] = current[self.history_fresh, None]
        self.history_fresh.zero_()
        return self.force_history.flatten(1)

    def dense_estimator_history_sym(self):
        dim = self.estimator_history_steps * 93
        return SymmetryTransform(torch.arange(dim), torch.ones(dim))

    def _select(self, ids):
        # All three anchors remain connected for the entire motion.
        self.slots[ids] = torch.arange(3, device=self.device)
        self.active[ids] = True
        self.releasing[ids] = False

    def _solve(self):
        raw = self.spring_gain * (self.anchor - self.point_positions_w())
        self.desired.copy_(limit_force(raw, per_point=30., total=60.))
        if not self.guidance_enabled:
            self.desired.zero_()

    def before_update(self):
        # Advance the continuous motion normally, without the inherited sparse
        # contact timer, root/yaw allocator, or direction projection.
        MotionTrackingCommand.before_update(self)
        self.update_reward_target()
        self._reference()
        self._solve()
        self.log_step += 1
        if self.log_step % 16 == 0:
            self.env.extra.update({
                'guidance/force_max_N': self.force.norm(dim=-1).max().item(),
                'guidance/force_budget_max_N': self.force.norm(dim=-1).sum(-1).max().item(),
                'guidance/anchor_error_m': (self.anchor - self.point_positions_w()).norm(dim=-1).mean().item(),
            })
