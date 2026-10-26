"""Kinematic prompt allocation; no inertia, ground contact or inverse dynamics."""
import torch


def allocate_jacobian_forces(jacobian, target, weights, active, previous,
                             force_penalty=1.0, smooth_penalty=1.0):
    """World point Jacobians [N,K,3,D]; target [N,D], weights [D]."""
    n, k, _, d = jacobian.shape
    mask = active[..., None].expand(n, k, 3).reshape(n, k * 3)
    g = jacobian.reshape(n, k * 3, d).transpose(1, 2) * mask[:, None]
    a = g * weights[None, :, None]
    b = target * weights
    h = a.transpose(1, 2) @ a
    h = h + (force_penalty + smooth_penalty) * torch.eye(k * 3, device=h.device, dtype=h.dtype)
    rhs = (a.transpose(1, 2) @ b[..., None]).squeeze(-1)
    rhs = rhs + smooth_penalty * previous.reshape(n, k * 3) * mask
    force = torch.linalg.solve(h, rhs[..., None]).squeeze(-1) * mask
    return force.reshape(n, k, 3)


def reconstruction(jacobian, force, target, weights):
    achieved = torch.einsum('nkcd,nkc->nd', jacobian, force)
    a, b = achieved * weights, target * weights
    residual = (a - b).norm(dim=-1)
    relative = residual / b.norm(dim=-1).clamp_min(1e-3)
    cosine = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-6)
    return achieved, residual, relative, cosine


def point_linear_jacobian(spatial_jacobian, offset_world):
    """Shift a COM spatial Jacobian to an application point: v_p=v_com+w×r."""
    angular = spatial_jacobian[..., 3:6, :].transpose(-1, -2)
    return spatial_jacobian[..., :3, :] + torch.cross(
        angular, offset_world[..., None, :].expand_as(angular), dim=-1).transpose(-1, -2)
