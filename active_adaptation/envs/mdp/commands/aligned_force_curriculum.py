"""Replay fixed8 aligned force prompts through the real physics and actor input."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from active_adaptation.envs.mdp import observation
from active_adaptation.utils.body_guidance import slew_force
from active_adaptation.utils.fixed8_force_encoder import load_fixed8_force_encoder
from active_adaptation.utils.math import quat_apply, quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from .motion_tracking import MotionTrackingCommand
from .passive_whole_body import PassiveWholeBodyCommand


class AlignedForceCurriculumCommand(PassiveWholeBodyCommand):
    def __init__(self, env, *, force_cache, latent_curriculum_start=.15,
                 latent_curriculum_end=.80, history_offsets=(0, 1, 2, 4, 8, 16),
                 force_encoder_path=None, **kwargs):
        allowed = {"root_ang_vel_history", "projected_gravity_history", "joint_pos_history",
                   "prev_actions", "sonic_joint_velocity", "latent_reference", "boot_indicator_state"}
        enabled = {k for k, v in env.cfg.observation.policy.items()
                   if not k.startswith("_") and (v is None or v.get("enabled", True))}
        if enabled - allowed:
            raise ValueError(f"Unaudited actor observation keys: {enabled - allowed}")
        paths = list(kwargs["dataset"]["mem_paths"])
        super().__init__(env, **kwargs)
        self.latent_curriculum_start = float(latent_curriculum_start)
        self.latent_curriculum_end = float(latent_curriculum_end)
        if not 0. <= self.latent_curriculum_start < self.latent_curriculum_end <= 1.:
            raise ValueError("Invalid curriculum bounds")
        self.latent_curriculum_alpha = 0.
        cache = Path(force_cache)
        metadata = json.loads((cache / "manifest.json").read_text())
        if metadata["version"] != "fixed8-aligned-observed-v1" or metadata["paths"] != paths:
            raise ValueError("Aligned force cache and training dataset must match")
        if abs(float(env.step_dt) - .02) > 1e-6:
            raise ValueError("Aligned force cache requires 50 Hz control")
        if self.dataset.source_indices is None or self.dataset.clamp_joint_targets:
            raise ValueError("Aligned replay requires source indices and unclamped joint references")
        dataset_root = Path(os.environ.get("MEMPATH", "dataset"))
        cached = []
        start = 1  # A shared zero row also handles the end of each clip.
        self.cache_offsets = torch.full((len(paths), max(ds.num_motions for ds in self.dataset.datasets)),
                                        -1, dtype=torch.long, device=self.device)
        for slot, (path, ds) in enumerate(zip(paths, self.dataset.datasets)):
            digest = hashlib.sha256((dataset_root / path / "meta_motion.json").read_bytes()).hexdigest()
            if digest != metadata["source_sha256"][slot]:
                raise ValueError(f"Source motion metadata changed for {path}")
            offsets = np.load(cache / f"offsets{slot}.npy")
            forces = np.load(cache / f"forces{slot}.npy")
            if len(offsets) != ds.num_motions or forces.shape[1:] != (8, 3):
                raise ValueError(f"Invalid timing or layout in force cache for {path}")
            absolute = np.where(offsets >= 0, offsets + start, -1)
            self.cache_offsets[slot, :len(absolute)] = torch.as_tensor(absolute, device=self.device)
            cached.append(torch.as_tensor(forces, device=self.device))
            start += len(forces)
        self.cached_force = torch.cat((cached[0].new_zeros(1, 8, 3), *cached))
        self.cached_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        points = metadata["points"]
        self.point_ids = torch.tensor([self.asset.body_names.index(p["body"]) for p in points],
                                      device=self.device)
        self.local = torch.tensor([p["offset"] for p in points], device=self.device)
        self.force = torch.zeros(self.num_envs, 8, 3, device=self.device)
        self.desired = torch.zeros_like(self.force)
        self.history_offsets = tuple(history_offsets)
        self.history = torch.zeros(self.num_envs, max(self.history_offsets) + 1, 24, device=self.device)
        self.force_encoder = None
        if force_encoder_path is not None:
            checkpoint = torch.load(force_encoder_path, map_location="cpu", weights_only=True)
            if (tuple(checkpoint["history_offsets"]) != self.history_offsets
                    or checkpoint["points"] != points
                    or checkpoint["solver_sha256"] != metadata["solver_sha256"]):
                raise ValueError("Force encoder must match the applied force layout, solver and history")
            self.force_encoder = load_fixed8_force_encoder(checkpoint).to(self.device)

    def step_schedule(self, progress, iter=0):
        x = (float(progress) - self.latent_curriculum_start) / (
            self.latent_curriculum_end - self.latent_curriculum_start)
        x = min(max(x, 0.), 1.)
        self.latent_curriculum_alpha = x * x * (3. - 2. * x)
        self.env.extra["curriculum/latent_autonomous_fraction"] = self.latent_curriculum_alpha

    def _lookup(self):
        # Cache row t is what an observer would see after applying interval t-1.
        future = self.t.long() + 1
        valid = (self.cached_start > 0) & (future < self.lengths) & self.guidance_enabled
        row = torch.where(valid, self.cached_start + future, 0)
        self.desired.copy_(self.cached_force[row])

    def reset(self, ids):
        MotionTrackingCommand.reset(self, ids)
        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)
        self.update_reward_target()
        self.force[ids] = 0.
        self.force_apply_buffer[ids] = 0.
        self.torque_apply_buffer[ids] = 0.
        self.history[ids] = 0.
        ds, clip = self.dataset.source_indices[ids].unbind(1)
        starts = self.cache_offsets[ds, clip]
        if (starts < 0).any():
            raise RuntimeError("Reference sampled from a motion missing in the force cache")
        self.cached_start[ids] = starts
        self._lookup()

    def before_update(self):
        MotionTrackingCommand.before_update(self)
        self.update_reward_target()
        self._lookup()
        current = quat_apply_inverse(self.asset.data.root_quat_w[:, None].expand(-1, 8, -1),
                                     self.force).flatten(1)
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = current
        self.log_step += 1
        if self.log_step % 16 == 0:
            norm = self.force.norm(dim=-1)
            self.env.extra.update({
                "aligned_force/mean_total_N": norm.sum(-1).mean().item(),
                "aligned_force/max_point_N": norm.max().item(),
                "aligned_force/active_contacts": (norm > .05).sum(-1).float().mean().item(),
            })

    def step(self, substep):
        self.force.copy_(slew_force(self.force, self.desired, self.env.physics_dt))
        self.force_apply_buffer.zero_()
        self.force_apply_buffer[:, self.point_ids] = self.force
        data = self.asset.data
        self.position_apply_buffer.copy_(data.body_pos_w)
        local = self.local.expand(self.num_envs, -1, -1)
        self.position_apply_buffer[:, self.point_ids] += quat_apply(data.body_quat_w[:, self.point_ids], local)
        self.asset.has_external_wrench = False

    @observation
    def aligned_force_history(self):
        history = self.history[:, self.history_offsets].flatten(1)
        if self.force_encoder is None:
            return history
        with torch.no_grad():
            return self.force_encoder(history)

    def aligned_force_history_sym(self):
        width = 32 if self.force_encoder is not None else 24 * len(self.history_offsets)
        return SymmetryTransform(torch.arange(width), torch.ones(width))

    @observation
    def latent_reference(self):
        return self.env.action_manager.current_latent

    def latent_reference_sym(self):
        return SymmetryTransform(torch.arange(32), torch.ones(32))

    @observation
    def sonic_joint_velocity(self):
        return self.asset.data.joint_vel[:, self.env.action_manager.joint_ids]

    def sonic_joint_velocity_sym(self):
        return self.env.action_manager.joint_symmetry_transform
