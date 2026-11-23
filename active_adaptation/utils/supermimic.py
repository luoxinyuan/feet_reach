"""Released SuperMimic actor and its feet_reach observation mapping."""
from pathlib import Path
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

SUPER_MIMIC_ROOT = Path(os.environ.get(
    "SUPER_MIMIC_ROOT", "/home/xl521/supermimic_force_transition"
)).expanduser().resolve()
SUPER_MIMIC_CODE = SUPER_MIMIC_ROOT / "code"
SUPER_MIMIC_CHECKPOINT = SUPER_MIMIC_ROOT / "pretrained/e11c_0220000/ckpt_e11c_0220000.pt"
SUPER_MIMIC_DEPLOY_CONFIG = SUPER_MIMIC_ROOT / "pretrained/e11c_0220000/deploy_config.json"

if not SUPER_MIMIC_CHECKPOINT.is_file() or not SUPER_MIMIC_DEPLOY_CONFIG.is_file():
    raise FileNotFoundError("Released SuperMimic checkpoint/deploy config are unavailable.")
sys.path.insert(0, str(SUPER_MIMIC_CODE))

from supermimic.rotations import heading_mat, mtv, quat_to_mat


class SuperMimicProprioceptionHistory:
    """Released actor's oldest-first, term-major 10-frame proprio history."""

    def __init__(self, n, device):
        self.values = [torch.zeros(n, 10, width, device=device) for width in (29, 29, 3, 3, 29)]
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.fresh[ids] = True
        for value in self.values:
            value[ids] = 0.

    def push(self, relative_q, qd, angular_velocity, gravity, last_action):
        for history, current in zip(
            self.values, (relative_q, qd, angular_velocity, gravity, last_action)
        ):
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
            history[self.fresh] = current[self.fresh, None]
        self.fresh.zero_()
        return torch.cat([x.flatten(1) for x in self.values], -1)


class ReleasedSuperMimic(nn.Module):
    """Minimal strict-load copy of the released 29-DOF tracking actor."""

    def __init__(self, checkpoint=SUPER_MIMIC_CHECKPOINT):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        p = state["policy"]
        self.encoder = nn.ModuleList([
            nn.Linear(680, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512),
            nn.Linear(512, 256), nn.Linear(256, 32),
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(962, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512),
            nn.Linear(512, 256), nn.Linear(256, 29),
        ])
        for i, layer in enumerate(self.encoder):
            layer.weight.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.bias"])
        for i, layer in enumerate(self.decoder):
            layer.weight.data.copy_(p[f"decoder.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"decoder.mlp.net.{2*i}.bias"])
        self.register_buffer("goal_scale", p["encoders.motion.scale"].clone())
        self.register_buffer("obs_scale", p["decoder.scale"].clone())
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def encode(self, goal):
        x = goal * self.goal_scale
        for layer in self.encoder[:-1]:
            x = F.silu(layer(x))
        return torch.tanh(self.encoder[-1](x))

    def forward(self, obs, goal):
        return self.decode(obs, self.encode(goal))

    def decode(self, obs, z):
        if z.shape[-1] != 32:
            raise ValueError("SuperMimic latent must be 32-dimensional")
        x = torch.cat((z, obs * self.obs_scale), dim=-1)
        for layer in self.decoder[:-1]:
            x = F.silu(layer(x))
        return self.decoder[-1](x)


def make_supermimic_goal(ref, env, robot, q0, qids, residual_q=None, residual_qd=None,
                         env_ids=None):
    """Create the release's 10-frame, 68-value-per-frame goal token."""
    rot = quat_to_mat(ref.root_quat_w)
    if env_ids is None:
        env_origins = env.scene.env_origins
        state_root_quat_w = robot.data.root_quat_w
        state_root_pos_w = robot.data.root_pos_w
    else:
        env_origins = env.scene.env_origins[env_ids]
        state_root_quat_w = robot.data.root_quat_w[env_ids]
        state_root_pos_w = robot.data.root_pos_w[env_ids]
    root_pos_w = ref.root_pos_w + env_origins[:, None, :]
    state_rot = quat_to_mat(state_root_quat_w)
    dp = mtv(state_rot[:, None], root_pos_w - state_root_pos_w[:, None])
    # Keep the transform calculation explicit as a frame-alignment assertion;
    # the released goal token uses root height and heading velocity, not XY.
    if not torch.isfinite(dp).all():
        raise RuntimeError("Non-finite root-relative motion goal.")
    q = ref.joint_pos[..., qids]
    qd = ref.joint_vel[..., qids]
    if residual_q is not None:
        q = q + residual_q[:, None]
    if residual_qd is not None:
        qd = qd + residual_qd[:, None]
    gravity_w = torch.zeros_like(ref.root_lin_vel_w)
    gravity_w[..., 2] = -1.0
    gravity_body = mtv(rot, gravity_w)
    ang_body = mtv(rot, ref.root_ang_vel_w)
    lin_heading = mtv(heading_mat(rot), ref.root_lin_vel_w)
    return torch.cat([q - q0, qd, gravity_body, ang_body, lin_heading,
                      ref.root_pos_w[..., 2:3]], dim=-1).flatten(1)


def current_supermimic_observation(robot, asset_joint_ids, q0, previous_action, history):
    q = robot.data.joint_pos[:, asset_joint_ids]
    qd = robot.data.joint_vel[:, asset_joint_ids]
    proprio = history.push(q - q0, qd, robot.data.root_ang_vel_b,
                           robot.data.projected_gravity_b, previous_action)
    return proprio


def make_reference_free_supermimic_goal(reference, robot, env_origins, q0,
                                         endpoint_correction=None):
    """Ten self-generated goal frames; no dataset or reference-motion access.

    Constant-velocity joint preview from the residual-owned reference, bounded
    by joint limits. Root goal retains observed height/translation/yaw rate,
    with upright orientation and zero roll/pitch rate. All root quantities are
    current proprioceptive measurements, never desired motion labels.
    """
    time = torch.arange(10, device=q0.device, dtype=q0.dtype)[None, :, None] * reference.dt
    proposal = reference.q[:, None] + time * reference.velocity[:, None]
    if endpoint_correction is not None:
        alpha = torch.linspace(0., 1., 10, device=q0.device, dtype=q0.dtype)[None, :, None]
        proposal = proposal + alpha * endpoint_correction[:, None]
    q = proposal.clamp(reference.lower[:, None], reference.upper[:, None])
    if endpoint_correction is None:
        qd = reference.velocity[:, None].expand_as(q).clone()
        qd[(proposal < reference.lower[:, None]) | (proposal > reference.upper[:, None])] = 0.
    else:
        previous_q = torch.cat((reference.q[:, None], q[:, :-1]), dim=1)
        qd = (q - previous_q) / reference.dt
    n = q.shape[0]
    gravity = torch.zeros(n, 10, 3, device=q.device); gravity[..., 2] = -1.
    angular = torch.zeros_like(gravity)
    angular[..., 2] = robot.data.root_ang_vel_w[:, 2:3]
    current_rot = quat_to_mat(robot.data.root_quat_w)
    linear = mtv(heading_mat(current_rot), robot.data.root_lin_vel_w)[:, None].expand(-1, 10, -1)
    height = (robot.data.root_pos_w[:, 2] - env_origins[:, 2])[:, None, None].expand(-1, 10, -1)
    return torch.cat([q-q0, qd, gravity, angular, linear, height], -1).flatten(1)


def make_endpoint_interpolation_supermimic_goal(robot, env_origins, q0,
                                                  current_q, endpoint_q,
                                                  lower, upper, dt):
    """Build a ten-frame straight-line preview from measured q to one endpoint.

    Intermediate joint targets are deterministic interpolation points; no
    velocity extrapolation or future motion reference is used. Root goal
    fields remain current-state measurements as in
    ``make_reference_free_supermimic_goal``.
    """
    alpha = torch.arange(1, 11, device=q0.device, dtype=q0.dtype)[None, :, None] / 10.
    delta = endpoint_q - current_q
    proposal = current_q[:, None] + alpha * delta[:, None]
    q = proposal.clamp(lower[:, None], upper[:, None])
    previous_q = torch.cat((current_q[:, None], q[:, :-1]), dim=1)
    qd = (q - previous_q) / dt
    n = q.shape[0]
    gravity = torch.zeros(n, 10, 3, device=q.device); gravity[..., 2] = -1.
    angular = torch.zeros_like(gravity)
    angular[..., 2] = robot.data.root_ang_vel_w[:, 2:3]
    current_rot = quat_to_mat(robot.data.root_quat_w)
    linear = mtv(heading_mat(current_rot), robot.data.root_lin_vel_w)[:, None].expand(-1, 10, -1)
    height = (robot.data.root_pos_w[:, 2] - env_origins[:, 2])[:, None, None].expand(-1, 10, -1)
    return torch.cat([q-q0, qd, gravity, angular, linear, height], -1).flatten(1)
