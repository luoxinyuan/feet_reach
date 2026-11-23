"""Frozen SuperMimic low-level tracker with a trainable joint residual input."""
import json
from pathlib import Path

import torch

from active_adaptation.envs.mdp.action import ActionManager
from active_adaptation.utils.sonic import ResidualReference
from active_adaptation.utils.supermimic import (
    ReleasedSuperMimic, SuperMimicProprioceptionHistory, SUPER_MIMIC_DEPLOY_CONFIG, current_supermimic_observation,
    make_supermimic_goal,
)
from active_adaptation.utils.symmetry import joint_space_symmetry


class SuperMimicResidualAction(ActionManager):
    """Track the motion reference with SuperMimic; PPO supplies bounded Δq residuals."""

    def __init__(self, env, reference_speed=2.0, max_reference_error=0.6,
                 deploy_config=None, **kwargs):
        super().__init__(env)
        if abs(env.step_dt - 0.02) > 1e-6:
            raise ValueError("SuperMimic release requires a 50 Hz policy rate.")
        config_path = Path(deploy_config) if deploy_config else SUPER_MIMIC_DEPLOY_CONFIG
        deploy = json.loads(config_path.read_text())
        self.policy_joint_names = list(deploy["joint_names"])
        if len(self.policy_joint_names) != 29 or set(self.policy_joint_names) != set(self.asset.joint_names):
            raise ValueError("SuperMimic deploy joint list must match the 29-DOF G1 asset by name.")
        self.joint_names = self.policy_joint_names
        self.joint_ids = [self.asset.joint_names.index(n) for n in self.policy_joint_names]
        dataset_names = self.env.command_manager.dataset.joint_names
        if set(dataset_names) != set(self.policy_joint_names):
            raise ValueError("Motion dataset and SuperMimic must both expose the same 29 joint names.")
        self.dataset_joint_ids = [dataset_names.index(n) for n in self.policy_joint_names]
        self.action_dim = 29
        self.q0 = torch.tensor(deploy["default_joint_pos"], dtype=torch.float32, device=self.device)
        self.action_scaling = torch.tensor(deploy["action_scale"], dtype=torch.float32, device=self.device)
        if self.q0.shape != (29,) or self.action_scaling.shape != (29,):
            raise ValueError("Invalid SuperMimic default pose/action scaling dimensions.")
        self.low_policy = ReleasedSuperMimic().to(self.device).eval()
        self.history = SuperMimicProprioceptionHistory(self.num_envs, self.device)
        limits = self.asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.reference = ResidualReference(self.num_envs, limits[..., 0], limits[..., 1],
                                           env.step_dt, reference_speed, max_reference_error)
        self.residual_origin = torch.zeros(self.num_envs, 29, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 29, device=self.device)
        # Keep the generic student observation interface used by the CEER/SONIC
        # task configs (`joint_pos_history` and `prev_actions`).
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        self.action_buf = torch.zeros(self.num_envs, 3, 29, device=self.device)
        self.low_action = torch.zeros_like(self.previous_action)
        self.target = self.asset.data.default_joint_pos.clone()
        self._goal_steps = torch.arange(1, 11, device=self.device)

    def symmetry_transforms(self):
        return joint_space_symmetry(self.asset, self.policy_joint_names)

    @torch.no_grad()
    def reset(self, env_ids):
        q = self.asset.data.joint_pos[:, self.joint_ids]
        self.reference.reset(env_ids, q)
        self.residual_origin[env_ids] = q[env_ids]
        self.history.reset(env_ids)
        self.previous_action[env_ids] = 0.0
        self.action_buf[env_ids] = 0.0
        self.low_action[env_ids] = 0.0
        self.target[env_ids] = self.asset.data.joint_pos[env_ids]

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            command = self.env.command_manager
            q = self.asset.data.joint_pos[:, self.joint_ids]
            residual_action = tensordict["action"].clamp(-10.0, 10.0)
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = residual_action
            self.reference.update(residual_action, q)
            residual_q = self.reference.q - self.residual_origin
            residual_qd = self.reference.velocity

            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            starts = command.t + 1
            ref = command.dataset.get_slice(None, starts, steps=self._goal_steps)
            goal = make_supermimic_goal(
                ref, self.env, self.asset, self.q0, self.dataset_joint_ids,
                residual_q=residual_q, residual_qd=residual_qd)
            self.low_action.copy_(self.low_policy(obs, goal))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action

        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class SuperMimicLatentResidualAction(SuperMimicResidualAction):
    """Add a PPO-predicted latent residual to the encoded 10-frame GT goal.

    The force command, future motion goal, proprioceptive history, and frozen
    SuperMimic decoder are identical to :class:`SuperMimicResidualAction`.
    Only the high-level action changes from 29 joint-reference residuals to a
    bounded 32-D residual added after the frozen motion encoder.
    """

    def __init__(self, env, latent_residual_scale=0.1, **kwargs):
        super().__init__(env, **kwargs)
        self.action_dim = 32
        self.latent_residual_scale = float(latent_residual_scale)
        if self.latent_residual_scale <= 0.0:
            raise ValueError("latent_residual_scale must be positive")
        self.action_buf = torch.zeros(self.num_envs, 3, 32, device=self.device)

    def symmetry_transforms(self):
        # Learned latent coordinates have no known exact left/right mapping.
        # The paired task disables symmetry augmentation.
        from active_adaptation.utils.symmetry import SymmetryTransform
        return SymmetryTransform(torch.arange(32), torch.ones(32))

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        self.action_buf[env_ids] = 0.0

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            command = self.env.command_manager
            action = tensordict["action"]
            if action.shape[-1] != 32:
                raise ValueError(f"Expected 32-D latent residual, got {action.shape[-1]}")
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action

            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            starts = command.t + 1
            ref = command.dataset.get_slice(None, starts, steps=self._goal_steps)
            goal = make_supermimic_goal(
                ref, self.env, self.asset, self.q0, self.dataset_joint_ids)
            base_latent = self.low_policy.encode(goal)
            latent_residual = self.latent_residual_scale * action.tanh()
            self.low_action.copy_(self.low_policy.decode(obs, base_latent + latent_residual))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action

        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class SuperMimicLatentIncrementCurriculumAction(SuperMimicLatentResidualAction):
    """Accumulate latent increments while curriculum removes future-GT bases.

    Each episode is seeded by the encoded GT window at its sampled start frame.
    During training the decoder base moves from the current future-GT latent to
    that fixed episode seed. The PPO action is always integrated as a persistent
    latent offset, so the final curriculum stage uses only seed + accumulated
    policy increments and never reads future motion for its action path.
    """

    def __init__(self, env, latent_increment_scale=0.025, **kwargs):
        super().__init__(env, **kwargs)
        self.latent_increment_scale = float(latent_increment_scale)
        if self.latent_increment_scale <= 0.0:
            raise ValueError("latent_increment_scale must be positive")
        self.seed_latent = torch.zeros(self.num_envs, 32, device=self.device)
        self.latent_offset = torch.zeros_like(self.seed_latent)
        self.current_latent = torch.zeros_like(self.seed_latent)

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        command = self.env.command_manager
        starts = torch.minimum(command.t[env_ids] + 1, command.lengths[env_ids] - 10)
        ref = command.dataset.get_slice(env_ids, starts, steps=self._goal_steps)
        goal = make_supermimic_goal(
            ref, self.env, self.asset, self.q0, self.dataset_joint_ids, env_ids=env_ids)
        seed = self.low_policy.encode(goal)
        self.seed_latent[env_ids] = seed
        self.latent_offset[env_ids] = 0.0
        self.current_latent[env_ids] = seed

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            command = self.env.command_manager
            action = tensordict["action"]
            if action.shape[-1] != 32:
                raise ValueError(f"Expected 32-D latent increment, got {action.shape[-1]}")
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
            self.latent_offset.add_(self.latent_increment_scale * action.tanh())

            alpha = float(getattr(command, "latent_curriculum_alpha", 0.0))
            if alpha < 1.0:
                starts = torch.minimum(command.t + 1, command.lengths - 10)
                ref = command.dataset.get_slice(None, starts, steps=self._goal_steps)
                goal = make_supermimic_goal(
                    ref, self.env, self.asset, self.q0, self.dataset_joint_ids)
                gt_latent = self.low_policy.encode(goal)
                base = torch.lerp(gt_latent, self.seed_latent, alpha)
            else:
                # Explicit final-stage fast path: no future motion slice/encoder.
                base = self.seed_latent

            latent = (base + self.latent_offset).clamp(-1.0, 1.0)
            self.current_latent.copy_(latent)
            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            self.low_action.copy_(self.low_policy.decode(obs, latent))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action

        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class SuperMimicDirectLatentAction(SuperMimicResidualAction):
    """Let PPO output the SuperMimic latent sent directly to its decoder.

    Unlike :class:`SuperMimicLatentResidualAction`, this action never builds
    or encodes a dataset motion goal.  The bounded 32-D PPO action is the
    complete latent input to the frozen decoder at every control step.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.action_dim = 32
        self.action_buf = torch.zeros(self.num_envs, 3, 32, device=self.device)
        self.last_latent = torch.zeros(self.num_envs, 32, device=self.device)

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        self.last_latent[env_ids] = 0.0

    def _decoder_latent(self, action):
        return action.tanh()

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            action = tensordict["action"]
            if action.shape[-1] != 32:
                raise ValueError(f"Expected 32-D direct latent, got {action.shape[-1]}")
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
            self.last_latent.copy_(self._decoder_latent(action))

            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            self.low_action.copy_(self.low_policy.decode(obs, self.last_latent))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action

        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class SuperMimicDirectLatentCurriculumAction(SuperMimicDirectLatentAction):
    """Fade reference assistance out of a directly predicted decoder latent."""

    def __init__(self, env, minimum_policy_weight=0.2, **kwargs):
        super().__init__(env, **kwargs)
        self.minimum_policy_weight = float(minimum_policy_weight)
        if not 0.0 < self.minimum_policy_weight < 1.0:
            raise ValueError("minimum_policy_weight must be between zero and one")
        self.current_latent = torch.zeros_like(self.last_latent)

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        self.current_latent[env_ids] = 0.0

    def _decoder_latent(self, action):
        proposal = super()._decoder_latent(action)
        # Actor feedback contains its own previous proposal throughout training.
        self.current_latent.copy_(proposal)
        command = self.env.command_manager
        progress_weight = command.latent_curriculum_alpha
        if progress_weight >= 1.0:
            return proposal

        starts = torch.minimum(command.t + 1, (command.lengths - 10).clamp_min(0))
        reference = command.dataset.get_slice(None, starts, steps=self._goal_steps)
        goal = make_supermimic_goal(
            reference, self.env, self.asset, self.q0, self.dataset_joint_ids)
        reference_latent = self.low_policy.encode(goal)
        policy_weight = self.minimum_policy_weight + (1.0 - self.minimum_policy_weight) * progress_weight
        return torch.lerp(reference_latent, proposal, policy_weight)


class ReferenceFreeSuperMimicResidualAction(SuperMimicResidualAction):
    """Residual-owned joint reference; frozen low level never reads GT motion."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        del self._goal_steps
        del self.dataset_joint_ids
        self.goal_interface = 'reference_free'
        self.last_goal = torch.zeros(self.num_envs, 680, device=self.device)

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            from active_adaptation.utils.supermimic import make_reference_free_supermimic_goal
            q = self.asset.data.joint_pos[:, self.joint_ids]
            action = tensordict['action'].clamp(-10., 10.)
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
            self.reference.update(action, q)
            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            self.last_goal.copy_(make_reference_free_supermimic_goal(
                self.reference, self.asset, self.env.scene.env_origins, self.q0))
            self.low_action.copy_(self.low_policy(obs, self.last_goal))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action
        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class EndpointPreviewSuperMimicResidualAction(ReferenceFreeSuperMimicResidualAction):
    """Reference-free residual with a policy-chosen future endpoint correction.

    The first 29 actions update the persistent current reference. The second
    29 choose a bounded correction to its 10-frame constant-velocity preview;
    that correction is interpolated smoothly from zero at the current frame
    to its full value at the last preview frame.
    """

    def __init__(self, env, preview_correction_scale=0.36, **kwargs):
        super().__init__(env, **kwargs)
        if preview_correction_scale <= 0.0:
            raise ValueError("preview_correction_scale must be positive")
        self.action_dim = 58
        self.preview_correction_scale = float(preview_correction_scale)
        self.action_buf = torch.zeros(self.num_envs, 3, self.action_dim, device=self.device)

    def symmetry_transforms(self):
        return joint_space_symmetry(self.asset, self.policy_joint_names).repeat(2)

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            from active_adaptation.utils.supermimic import make_reference_free_supermimic_goal
            q = self.asset.data.joint_pos[:, self.joint_ids]
            action = tensordict['action'].clamp(-10., 10.)
            if action.shape[-1] != self.action_dim:
                raise ValueError(f"Expected {self.action_dim}-D residual/preview action, got {action.shape[-1]}")
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
            self.reference.update(action[:, :29], q)
            endpoint_correction = self.preview_correction_scale * action[:, 29:].tanh()
            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            self.last_goal.copy_(make_reference_free_supermimic_goal(
                self.reference, self.asset, self.env.scene.env_origins, self.q0,
                endpoint_correction=endpoint_correction))
            self.low_action.copy_(self.low_policy(obs, self.last_goal))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action
        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class EndpointInterpolationSuperMimicResidualAction(ReferenceFreeSuperMimicResidualAction):
    """Predict one joint endpoint and interpolate from measured q.

    Each 29-D policy action directly predicts an endpoint offset from measured
    q. The frozen SuperMimic receives ten joint targets on a straight line
    from measured current pose to that endpoint. It does not receive a
    persistent-reference update, constant-velocity extrapolation, or ten
    independently predicted frames.
    """

    def __init__(self, env, endpoint_offset_scale=0.36, **kwargs):
        super().__init__(env, **kwargs)
        if endpoint_offset_scale <= 0.0:
            raise ValueError("endpoint_offset_scale must be positive")
        self.endpoint_offset_scale = float(endpoint_offset_scale)

    @torch.no_grad()
    def __call__(self, tensordict, substep):
        if substep == 0:
            from active_adaptation.utils.supermimic import make_endpoint_interpolation_supermimic_goal
            q = self.asset.data.joint_pos[:, self.joint_ids]
            action = tensordict['action'].clamp(-10., 10.)
            if action.shape[-1] != 29:
                raise ValueError(f"Expected 29-D endpoint action, got {action.shape[-1]}")
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
            endpoint_q = (q + self.endpoint_offset_scale * action.tanh()).clamp(
                self.reference.lower, self.reference.upper)
            obs = current_supermimic_observation(
                self.asset, self.joint_ids, self.q0, self.previous_action, self.history)
            self.last_goal.copy_(make_endpoint_interpolation_supermimic_goal(
                self.asset, self.env.scene.env_origins, self.q0, q, endpoint_q,
                self.reference.lower, self.reference.upper, self.reference.dt))
            self.low_action.copy_(self.low_policy(obs, self.last_goal))
            self.previous_action.copy_(self.low_action)
            self.target[:, self.joint_ids] = self.q0 + self.action_scaling * self.low_action
        self.asset.set_joint_position_target(self.target)
        self.asset.write_data_to_sim()


class LatentEndpointSuperMimicResidualAction(ReferenceFreeSuperMimicResidualAction):
    """32-D latent residual endpoint; causal 10-control-frame interpolation.

    Zero residual decodes a self-generated hold-reset-pose goal. Residual
    endpoints cover the full bounded latent range; interpolate 1/10 of the
    remaining distance per 50Hz update (receding horizon). No future GT access.
    The decoder takes ONE latent, not a sequence of ten latent goal tokens.
    """
    def __init__(self, env, latent_scale=2.0, interpolation_frames=10, **kwargs):
        super().__init__(env, **kwargs)
        self.action_dim=32;self.latent_scale=float(latent_scale)
        self.interpolation_frames=int(interpolation_frames)
        if self.interpolation_frames<1:raise ValueError(interpolation_frames)
        self.action_buf=torch.zeros(self.num_envs,3,32,device=self.device)
        self.latent_residual=torch.zeros(self.num_envs,32,device=self.device)
        self.latent_endpoint=torch.zeros_like(self.latent_residual)
        self.last_latent=torch.zeros_like(self.latent_residual)
        self.goal_interface='reference_free_latent_endpoint'

    def symmetry_transforms(self):
        from active_adaptation.utils.symmetry import SymmetryTransform
        return SymmetryTransform(torch.arange(32),torch.ones(32))

    def reset(self, ids):
        super().reset(ids)
        self.latent_residual[ids]=0;self.latent_endpoint[ids]=0;self.last_latent[ids]=0

    @torch.no_grad()
    def __call__(self,tensordict,substep):
        if substep==0:
            from active_adaptation.utils.supermimic import make_reference_free_supermimic_goal
            q=self.asset.data.joint_pos[:,self.joint_ids]
            action=tensordict['action'].clamp(-10,10)
            self.action_buf=self.action_buf.roll(1,dims=1);self.action_buf[:,0]=action
            self.latent_endpoint.copy_(self.latent_scale*action.tanh())
            self.latent_residual.add_((self.latent_endpoint-self.latent_residual)/self.interpolation_frames)
            self.last_goal.copy_(make_reference_free_supermimic_goal(self.reference,self.asset,self.env.scene.env_origins,self.q0))
            base=self.low_policy.encode(self.last_goal)
            self.last_latent.copy_((base+self.latent_residual).clamp(-1.,1.))
            obs=current_supermimic_observation(self.asset,self.joint_ids,self.q0,self.previous_action,self.history)
            self.low_action.copy_(self.low_policy.decode(obs,self.last_latent))
            self.previous_action.copy_(self.low_action)
            self.target[:,self.joint_ids]=self.q0+self.action_scaling*self.low_action
        self.asset.set_joint_position_target(self.target);self.asset.write_data_to_sim()
