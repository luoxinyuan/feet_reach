"""Target-private spring fitting and bounded physical force; SI units."""
import torch


def fit_spring(path, dt, damping=8.0, mass=1.0, minimum=40., maximum=160.):
    """Solve min_k ||k(goal-x)-b*v-m*a||² + .01(k-80)², box constrained.

    path: [batch, time, hands, xyz]. This is a Cartesian surrogate, not a
    proof that an arbitrary whole-body displacement is achievable with EE force.
    """
    velocity = (path[:, 2:] - path[:, :-2]) / (2 * dt)
    acceleration = (path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]) / dt**2
    error = path[:, -1:] - path[:, 1:-1]
    desired = damping * velocity + mass * acceleration
    numerator = (error * desired).sum(dim=(1, 3)) + .01 * 80.
    denominator = error.square().sum(dim=(1, 3)) + .01
    return (numerator / denominator).clamp(minimum, maximum)


def spring_force(anchor, position, velocity, stiffness, active, damping=8., limit=60.):
    force = stiffness[..., None] * (anchor-position) - damping * velocity
    force = force * (limit / force.norm(dim=-1, keepdim=True).clamp_min(limit))
    return force * active[..., None]
