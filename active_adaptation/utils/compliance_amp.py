"""Shared motion-only features for reference/rollout adversarial motion prior."""
import torch
def quat_apply_inverse(q,v):
    xyz=q[...,1:];cross=torch.cross(xyz,v,dim=-1)
    return v-2*q[...,:1]*cross+2*torch.cross(xyz,cross,dim=-1)
AMP_BODIES=['left_hand_mimic','right_hand_mimic','head_mimic','left_ankle_roll_link','right_ankle_roll_link']

def motion_features(q,qd,root_pos,root_quat,lin,ang,points):
    gravity=torch.zeros_like(root_pos);gravity[...,2]=-1
    local_points=quat_apply_inverse(root_quat[...,None,:].expand(*points.shape[:-1],4),points-root_pos[...,None,:])
    return torch.cat((root_pos[...,2:3],quat_apply_inverse(root_quat,gravity),
        quat_apply_inverse(root_quat,lin),quat_apply_inverse(root_quat,ang),q,qd,local_points.flatten(-2)),-1)

def pair_features(current,following):
    return torch.cat((current,following),-1)

def style_reward(score):
    # Least-squares AMP: expert +1, policy -1. Bounded per-second reward.
    return (1.-.25*(score-1.).square()).clamp(0.,1.)
