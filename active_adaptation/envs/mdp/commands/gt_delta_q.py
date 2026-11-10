"""Reference-to-reference joint increments: feedforward oracle control."""
import hashlib
import json
from pathlib import Path
import torch
from .sonic_compliance import SonicComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.sonic import JOINT_NAMES
from active_adaptation.utils.symmetry import SymmetryTransform


class GTDeltaQCommand(SonicComplianceCommand):
    def __init__(self, env, *args, guidance_cache, delta_q_scale=0.04,
                 allow_dataset_override=False, **kwargs):
        super().__init__(env, *args, **kwargs)
        # Use the identical packed motion corpus; do not load its force tensors.
        manifest = json.loads((Path(guidance_cache) / 'manifest.json').read_text())
        if not manifest['complete'] or abs(manifest['dt'] - env.step_dt) > 1e-8:
            raise ValueError('Requires complete, matching-rate motion corpus')
        if (not allow_dataset_override
                and list(env.cfg.command.dataset.mem_paths) != manifest['dataset_paths']):
            raise ValueError('Baseline dataset mismatch')
        for source, digest in zip(manifest['dataset_paths'], manifest['dataset_meta_sha256']):
            if hashlib.sha256((Path(source) / 'meta_motion.json').read_bytes()).hexdigest() != digest:
                raise ValueError('Dataset metadata changed')
        self.reference_target_ids = [self.dataset.joint_names.index(n) for n in JOINT_NAMES]
        self.delta_q_scale = float(delta_q_scale)
        if self.delta_q_scale <= 0:
            raise ValueError('delta_q_scale must be positive')
        self.delta_q_buffer = torch.zeros(self.num_envs, len(JOINT_NAMES), device=self.device)
        self.force_apply_world = False

    def _reference(self):
        pass

    def _select(self, ids):
        pass

    def _solve(self):
        nxt = self.dataset.get_slice(None, torch.minimum(self.t + 1, self.lengths - 1), 1)
        self.delta_q_buffer.copy_(nxt.joint_pos[:, 0, self.reference_target_ids].float()
                                 - self._motion.joint_pos[:, 0, self.reference_target_ids].float())
        self.delta_q_buffer[self.t >= self.lengths - 1] = 0

    def before_update(self):
        MotionTrackingCommand.before_update(self)
        self.update_reward_target()
        self._solve()
        self.log_step += 1
        if self.log_step % 16 == 0:
            self.env.extra.update({
                'prompt/delta_q_rms_rad': self.delta_q_buffer.square().mean().sqrt().item(),
                'prompt/physical_force_max_N': self.force_apply_buffer.norm(dim=-1).max().item(),
                'tracking/reference_joint_mae_rad': (self.env.action_manager.reference.q - self._motion.joint_pos[:, 0, self.reference_target_ids]).abs().mean().item(),
                'tracking/joint_mae_rad': (self.asset.data.joint_pos[:, self.all_joint_idx_asset] - self._motion.joint_pos[:, 0, self.all_joint_idx_dataset]).abs().mean().item(),
            })

    def step(self, substep):
        self.force_apply_buffer.zero_()
        self.torque_apply_buffer.zero_()
        self.asset.has_external_wrench = False

    @observation
    def gt_delta_q(self):
        # Fixed invertible scaling, no clipping and no running normalization.
        return self.delta_q_buffer / self.delta_q_scale

    def gt_delta_q_sym(self):
        return SymmetryTransform(torch.arange(len(JOINT_NAMES)), torch.ones(len(JOINT_NAMES)))
