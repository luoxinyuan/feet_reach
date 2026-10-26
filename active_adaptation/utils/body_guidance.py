"""Geometric intent allocation, without inertia or contact dynamics."""
import torch


def allocate_forces(local, arms, desired_force, desired_yaw, active, previous):
    # Three slots, nine scalar unknowns. Weighted least-squares compromise.
    n=local.shape[0]; mask=active[...,None].expand(-1,-1,3).reshape(n,9).float()
    identity=torch.eye(3,device=local.device,dtype=local.dtype).repeat(1,3).expand(n,-1,-1)
    yaw=torch.stack((-arms[...,1],arms[...,0],torch.zeros_like(arms[...,0])),-1).reshape(n,1,9)
    # 0.3 m characteristic arm makes yaw residual comparable to force residual.
    a=torch.cat((identity,yaw/.3),1)*mask[:,None]
    goal=torch.cat((desired_force,desired_yaw[:,None]/.3),-1)
    h=2.*torch.eye(9,device=local.device).expand(n,-1,-1)+4.*a.transpose(1,2)@a
    b=(local+previous).reshape(n,9)*mask+4.*(a.transpose(1,2)@goal[...,None]).squeeze(-1)
    return (torch.linalg.solve(h,b[...,None]).squeeze(-1)*mask).reshape(n,3,3)


def limit_force(force, per_point=30., total=60.):
    # Uniform scaling preserves the net-force and yaw directions.
    norms=force.norm(dim=-1)
    scale=torch.minimum(per_point/norms.amax(-1).clamp_min(per_point),total/norms.sum(-1).clamp_min(total))
    return force*scale[:,None,None]


def consistent_force(force, arms, desired_force, desired_yaw, active):
    """Project onto non-opposing translation/yaw intent halfspaces.

    Homogeneous constraints always admit zero. Opposed/infeasible layouts can
    lose strength rather than inventing an opposing guidance command.
    """
    mask=active[...,None].expand_as(force)
    a=desired_force[:,None,:].expand_as(force)*mask
    b=torch.stack((-arms[...,1],arms[...,0],torch.zeros_like(arms[...,0])),-1)*desired_yaw[:,None,None]*mask
    for _ in range(8):
        for direction in (a,b):
            dot=(force*direction).sum((1,2))
            correction=(-dot).clamp_min(0)/direction.square().sum((1,2)).clamp_min(1e-9)
            force=force+correction[:,None,None]*direction
    bad=((force*a).sum((1,2)) < -1e-4)|((force*b).sum((1,2)) < -1e-4)
    return force.masked_fill(bad[:,None,None],0.)


def slew_force(previous, desired, dt, rate=150.):
    # Common interpolation factor preserves convex per-point and total budgets.
    delta=desired-previous
    alpha=(rate*dt/delta.norm(dim=-1).amax(-1).clamp_min(1e-9)).clamp_max(1.)
    return previous+alpha[:,None,None]*delta
