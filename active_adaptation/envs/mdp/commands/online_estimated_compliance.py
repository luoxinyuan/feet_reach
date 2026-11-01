"""Online Jacobian prompts with causal CEER-style force adaptation."""
import torch
from .jacobian_compliance import JacobianComplianceCommand
from .dense_link_compliance import DenseLinkComplianceCommand


class OnlineEstimatedComplianceCommand(JacobianComplianceCommand):
    # Reuse the tested GT labels and causal proprioception history only;
    # no offline force-cache constructor or force application is inherited.
    dense_force_gt = DenseLinkComplianceCommand.dense_force_gt
    dense_force_gt_sym = DenseLinkComplianceCommand.dense_force_gt_sym
    dense_estimator_history = DenseLinkComplianceCommand.dense_estimator_history
    dense_estimator_history_sym = DenseLinkComplianceCommand.dense_estimator_history_sym

    def __init__(self, env, *args, estimator_history=10, expected_force_points=3, **kwargs):
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

    @property
    def point_force_buffer(self):
        # Canonical point order, independent of randomized active slot order.
        return self.force_apply_buffer[:, self.point_ids]

    def reset(self, ids):
        super().reset(ids)
        self.force_history[ids] = 0.
        self.history_fresh[ids] = True
