"""Reuse force prompts/rewards while exposing only the student's own reference."""
import torch
from .passive_whole_body import PassiveWholeBodyCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.symmetry import SymmetryTransform

class SonicComplianceCommand(PassiveWholeBodyCommand):
    def __init__(self, env, *args, **kwargs):
        allowed = {'root_ang_vel_history', 'projected_gravity_history',
                   'joint_pos_history', 'prev_actions', 'upper_body_forces_b',
                   'sonic_reference', 'sonic_joint_velocity'}
        enabled = {key for key, value in env.cfg.observation.policy.items()
                   if not key.startswith('_') and (value is None or value.get('enabled', True))}
        if enabled - allowed:
            raise ValueError(f'Unaudited student observation terms: {enabled - allowed}')
        super().__init__(env, *args, **kwargs)

    @observation
    def sonic_reference(self):
        manager = self.env.action_manager
        ref = manager.reference
        actual = self.asset.data.joint_pos[:, manager.joint_ids]
        return torch.cat((ref.q, ref.q - actual, ref.velocity), -1)

    def sonic_reference_sym(self):
        # This task explicitly disables symmetry augmentation.
        return SymmetryTransform(torch.arange(87), torch.ones(87))

    @observation
    def sonic_joint_velocity(self):
        return self.asset.data.joint_vel[:, self.env.action_manager.joint_ids]

    def sonic_joint_velocity_sym(self):
        return self.env.action_manager.symmetry_transforms()
