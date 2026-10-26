"""Reduce point forces to mechanically equivalent per-link wrenches."""
import torch


def reduce_point_forces(force, point_position, body_ids, body_position):
    """Force slots may share a link. Moments are only r×f, not torque commands."""
    index = body_ids[..., None].expand_as(force)
    total_force = torch.zeros_like(body_position).scatter_add_(1, index, force)
    origin = body_position.gather(1, index)
    moments = torch.cross(point_position - origin, force, dim=-1)
    total_moment = torch.zeros_like(body_position).scatter_add_(1, index, moments)
    return total_force, total_moment
