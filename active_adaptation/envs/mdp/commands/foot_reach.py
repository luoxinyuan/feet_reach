"""Fixed-wall left-foot reaching through the existing low-level PPO pipeline."""
import torch

from active_adaptation.envs.mdp import observation, reward
from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand
from active_adaptation.utils.math import quat_apply, quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry


class WallFootReachCommand(MotionTrackingCommand):
    supports_symmetry_augmentation = False

    def __init__(self, env, *args, sole_offset=(0., 0., -.03540915), foot_force=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        if self.dataset.position_z_offset != 0 or self.dataset.clamp_joint_targets:
            raise ValueError("Fixed-contact references require zero z offset and no joint target clamping")
        self.sole_offset = torch.tensor(sole_offset, device=self.device, dtype=torch.float32)
        self.left_motion = self.dataset.body_names.index("left_ankle_roll_link")
        self.left_asset = self.asset.body_names.index("left_ankle_roll_link")
        self.right_asset = self.asset.body_names.index("right_ankle_roll_link")
        self.right_motion = self.dataset.body_names.index("right_ankle_roll_link")
        self.hand_asset = self.asset.body_names.index("left_hand_mimic")
        self.hand_motion = self.dataset.body_names.index("left_hand_mimic")
        self._external_target_b = None
        from active_adaptation.utils.foot_force import FootForceRamp
        force_cfg = dict(foot_force or {})
        self.foot_force_enabled = force_cfg.pop('enabled', False)
        self.foot_force_train_only = force_cfg.pop('train_only', True)
        self.foot_force_ramp = FootForceRamp(self.num_envs, self.device, **force_cfg)
        shape = (self.num_envs, len(self.asset.body_names), 3)
        self.force_apply_buffer = torch.zeros(shape, device=self.device)
        self.torque_apply_buffer = torch.zeros_like(self.force_apply_buffer)
        self.position_apply_buffer = torch.zeros_like(self.force_apply_buffer)
        self.force_apply_world = False
        self.zero_init_prob = 0.  # Sample across each trajectory, not only its anchor.

    def step(self, substep):
        enabled = self.foot_force_enabled and (self.env.training or not self.foot_force_train_only)
        self.force_apply_world = bool(enabled)
        self.force_apply_buffer.zero_()
        if not enabled:
            self.foot_force_ramp.force.zero_()
            return
        if substep == 0:
            self.foot_force_ramp.advance()
            magnitudes = self.foot_force_ramp.force.norm(dim=-1)
            self.env.extra['force/foot_mean_N'] = magnitudes.mean().item()
            self.env.extra['force/foot_max_N'] = magnitudes.max().item()
        self.force_apply_buffer[:, self.left_asset] = self.foot_force_ramp.force
        # World-frame forces at the actual moving sole, not at the ankle COM.
        # PhysX accounts for the lever arm; explicit additional torque is zero.
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.position_apply_buffer[:, self.left_asset] += quat_apply(
            self.asset.data.body_quat_w[:, self.left_asset], self.sole_offset.expand(self.num_envs, -1))
        self.asset.has_external_wrench = False

    @observation
    def foot_external_force_b(self):
        return quat_apply_inverse(self.asset.data.root_quat_w, self.foot_force_ramp.force)

    def foot_external_force_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., -1., 1.])

    def sample_init_robot(self, env_ids, motion, lift_height=0.):
        # The dataset already contains the physical floor and wall coordinates.
        return super().sample_init_robot(env_ids, motion, lift_height=0.)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.foot_force_ramp.reset(env_ids)
        self.force_apply_buffer[env_ids] = 0
        self.torque_apply_buffer[env_ids] = 0
        # sample_init changed the clip/time; refresh targets without advancing it.
        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)
        self.update_reward_target()

    def before_update(self):
        super().before_update()
        self.update_reward_target()

    def update_reward_target(self):
        # Never let teacher/student root targets drift with the live robot: the
        # wall is fixed in each environment and cannot follow a drifting target.
        self.reward_keypoints_w = self._motion.body_pos_w[:, 0] + self.env.scene.env_origins[:, None]
        self.reward_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
        self.reward_root_quat_w = self._motion.root_quat_w[:, 0]

    def target_sole_world(self):
        if self._external_target_b is not None:
            return self.asset.data.root_pos_w + quat_apply(self.asset.data.root_quat_w, self._external_target_b)
        foot = self._motion.body_pos_w[:, 0, self.left_motion]
        quat = self._motion.body_quat_w[:, 0, self.left_motion]
        return foot + quat_apply(quat, self.sole_offset.expand(self.num_envs, -1)) + self.env.scene.env_origins

    def set_target_foot_pos_b(self, target):
        """Deployment command: desired left sole xyz in the live root frame.

        None restores motion-dataset commands. This changes student commands,
        not privileged teacher joint labels; use it for student evaluation.
        """
        if target is None:
            self._external_target_b = None
            return
        target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        if target.shape == (3,):
            target = target.expand(self.num_envs, -1)
        if target.shape != (self.num_envs, 3) or not torch.isfinite(target).all():
            raise ValueError("Target must be finite xyz or [num_envs, 3]")
        self._external_target_b = target.clone()

    @observation
    def target_foot_pos_b(self):
        return quat_apply_inverse(self.asset.data.root_quat_w,
                                  self.target_sole_world() - self.asset.data.root_pos_w)

    def target_foot_pos_b_sym(self):
        # No meaningful left/right mirror exists for this fixed unilateral task.
        # PPO symmetry augmentation is explicitly disabled in the task launcher.
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def target_joint_pos_obs(self):
        # Current 29-joint target only: future targets would leak trajectory
        # choices that cannot be inferred from a single xyz command.
        return self._motion.joint_pos[:, 0, self.joint_idx_motion]

    def target_joint_pos_obs_sym(self):
        return joint_space_symmetry(self.asset, [self.dataset.joint_names[i] for i in self.joint_idx_motion.tolist()])

    @reward
    def swing_foot_tracking(self):
        current = self.asset.data.body_pos_w[:, self.left_asset] + quat_apply(
            self.asset.data.body_quat_w[:, self.left_asset], self.sole_offset.expand(self.num_envs,-1))
        error = torch.linalg.vector_norm(self.target_sole_world()-current, dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return torch.exp(-error / .06)

    @reward
    def support_foot_tracking(self):
        target = self._motion.body_pos_w[:,0,self.right_motion]+self.env.scene.env_origins
        error = torch.linalg.vector_norm(target-self.asset.data.body_pos_w[:,self.right_asset],dim=-1,keepdim=True)
        return torch.exp(-error/.025)

    @reward
    def wall_hand_tracking(self):
        target = self._motion.body_pos_w[:,0,self.hand_motion]+self.env.scene.env_origins
        error = torch.linalg.vector_norm(target-self.asset.data.body_pos_w[:,self.hand_asset],dim=-1,keepdim=True)
        return torch.exp(-error/.025)

    def debug_draw(self):
        if hasattr(self.env, "debug_draw"):
            self.env.debug_draw.point(self.target_sole_world(),color=(0.,1.,.5,1.),size=12.)
