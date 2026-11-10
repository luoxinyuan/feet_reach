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
