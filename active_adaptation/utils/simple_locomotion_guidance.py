"""Simple human-like waist pull and torso twist, without a dynamics model."""
import torch


def yaw_from_quat(q):
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def guidance(position, quaternion, anchor, target_yaw, hold,
             stiffness=60., force_limit=30., yaw_stiffness=8., torque_limit=4.):
    force=stiffness*(anchor-position)
    force=force.clone();force[...,2]=0.
    force*=force_limit/force.norm(dim=-1,keepdim=True).clamp_min(force_limit)
    delta=target_yaw-yaw_from_quat(quaternion)
    error=torch.atan2(torch.sin(delta),torch.cos(delta))
    torque=torch.zeros_like(force);torque[...,2]=(yaw_stiffness*error).clamp(-torque_limit,torque_limit)
    force=force.masked_fill(hold[...,None],0.)
    torque=torque.masked_fill(hold[...,None],0.)
    return force,torque
