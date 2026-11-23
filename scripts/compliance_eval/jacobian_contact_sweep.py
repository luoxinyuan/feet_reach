"""Enumerate fixed-keypoint force allocations at a live rollout snapshot."""
from itertools import combinations

import torch

from active_adaptation.utils.body_guidance import consistent_force, limit_force
from active_adaptation.utils.contact_observability import reconstruction_score
from active_adaptation.utils.jacobian_guidance import (
    allocate_jacobian_forces, point_linear_jacobian, reconstruction,
)


@torch.no_grad()
def best_allocations(command, ids, subsets, geometry=None):
    """Score candidate sets after the allocator's direction correction and force caps."""
    targets = command.generalized_target[ids]
    weights = command.prompt_weights
    if geometry is None:
        jacobians = command.point_jacobians[ids]
        arms = (command.asset.data.body_pos_w[ids[:, None], command.point_ids]
                - command.asset.data.root_pos_w[ids, None])
    else:
        jacobians, arms = geometry
    net_force, yaw = command._intent()
    results = []
    for contacts in subsets:
        n, m, k = len(ids), *contacts.shape
        jac = jacobians[:, contacts].flatten(0, 1)
        lever = arms[:, contacts].flatten(0, 1)
        target = targets.repeat_interleave(m, dim=0)
        active = torch.ones((n*m, k), dtype=torch.bool, device=jac.device)
        forces = allocate_jacobian_forces(jac, target, weights, active,
                                          lever.new_zeros(lever.shape),
                                          command.force_penalty, command.smooth_penalty)
        forces = limit_force(consistent_force(forces, lever, net_force[ids].repeat_interleave(m, dim=0),
                                              yaw[ids].repeat_interleave(m), active,
                                              match_net_force=command.match_net_force))
        _, _, errors, _ = reconstruction(jac, forces, target, weights)
        best = errors.reshape(n, m).argmin(-1)
        rows = torch.arange(n, device=jac.device)
        best_forces = forces.reshape(n, m, k, 3)[rows, best]
        norms = best_forces.norm(dim=-1)
        results.append((errors.reshape(n, m)[rows, best], norms.sum(-1),
                        (norms > .05).sum(-1), contacts[best]))
    return results, (targets*weights).norm(dim=-1)


def contact_subsets(command):
    return [torch.tensor(list(combinations(range(len(command.point_ids)), k)),
                         device=command.point_ids.device) for k in (1, 2, 3)]


def offset_point_geometry(command, ids, body_ids, local_offsets):
    """Evaluate a link-local atlas using the same world/root-origin convention."""
    from active_adaptation.utils.math import quat_apply

    asset = command.asset
    body_quat = asset.data.body_quat_w[ids[:, None], body_ids]
    point = (asset.data.body_pos_w[ids[:, None], body_ids]
             + quat_apply(body_quat, local_offsets.expand(len(ids), -1, -1)))
    spatial = asset.root_physx_view.get_jacobians()[ids[:, None], body_ids]
    jac = point_linear_jacobian(spatial, point - asset.data.body_com_pos_w[ids[:, None], body_ids])
    lever = point - asset.data.root_pos_w[ids, None]
    eye = torch.eye(3, device=jac.device, dtype=jac.dtype)
    axes = eye.expand(len(ids), len(body_ids), -1, -1)
    jac[..., :3] = axes
    jac[..., 3:6] = torch.cross(axes, lever.unsqueeze(-2).expand_as(axes), dim=-1).transpose(-1, -2)
    return jac, lever


def reference_transition(command, ids):
    from active_adaptation.utils.math import quat_conjugate, quat_mul

    ref = command.dataset.get_slice(ids, command.t[ids], 2)
    relative = quat_mul(ref.root_quat_w[:, 1], quat_conjugate(ref.root_quat_w[:, 0]))
    relative = relative * torch.where(relative[:, :1] < 0, -1., 1.)
    axis = relative[:, 1:]
    norm = axis.norm(dim=-1, keepdim=True)
    rotation = axis * (2 * torch.atan2(norm, relative[:, :1]) / norm.clamp_min(1e-8))
    delta = command.generalized_target[ids].new_zeros((len(ids), 6 + command.asset.num_joints))
    delta[:, :3] = ref.root_pos_w[:, 1] - ref.root_pos_w[:, 0]
    delta[:, 3:6] = rotation
    joint_ids = command.all_joint_idx_dataset
    delta[:, 6 + command.all_joint_idx_asset] = ref.joint_pos[:, 1, joint_ids] - ref.joint_pos[:, 0, joint_ids]
    return delta


def best_motion_reconstruction(jacobians, delta, subsets):
    """Linearized displacement recoverability, using the same per-point observation model for both layouts."""
    scale = delta.new_full((delta.shape[-1],), .3)
    scale[:3] = .1
    motion = delta / scale
    scores = []
    for contacts in subsets:
        n, m, k = len(delta), *contacts.shape
        jac = jacobians[:, contacts].reshape(n*m, k, 3, -1) * scale
        target = motion.repeat_interleave(m, dim=0).unsqueeze(1)
        score = reconstruction_score(jac, target)
        scores.append(score.reshape(n, m).amin(dim=-1))
    return torch.stack(scores, dim=-1), motion.norm(dim=-1)
