"""Independent next-frame anchor springs, without wrench allocation."""
import torch

from .expanded_spring_compliance import ExpandedSpringComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.utils.body_guidance import limit_force


class DirectKeypointSpringCommand(ExpandedSpringComplianceCommand):
    def __init__(self, env, *args, spring_gain=80., **kwargs):
        super().__init__(env, *args, **kwargs)
        if len(self.point_names) != 3:
            raise ValueError('Direct root/EE springs require exactly three points.')
        if spring_gain <= 0:
            raise ValueError('spring_gain must be positive.')
        self.spring_gain = float(spring_gain)

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
