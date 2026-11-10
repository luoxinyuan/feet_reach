"""Independent translation prompt and bounded two-hand joint/yaw allocation."""
import torch

def allocate_independent(root_delta, yaw_delta, hand_jacobian, joint_target,
                         previous_hands, dt=.02, pelvis_gain=1190.6272466177975,
                         yaw_gain=83., yaw_gate_low=.05, yaw_gate_high=.3,
                         force_penalty=1., smooth_penalty=1.):
    # hand_jacobian: [N,2,3,1+upper_joint_count]; first column is world yaw.
    pelvis=pelvis_gain*root_delta
    pelvis*= (30./pelvis.norm(dim=-1,keepdim=True).clamp_min(30.))
    speed=yaw_delta.abs()/dt
    x=((speed-yaw_gate_low)/(yaw_gate_high-yaw_gate_low)).clamp(0,1)
    gate=x*x*(3-2*x)
    yaw_target=(yaw_gain*yaw_delta).clamp(-4,4)
    target=torch.cat((yaw_target[:,None],joint_target),-1)
    weights=torch.full_like(target,1/.3)
    weights[:,0]=(2/.3)*gate.sqrt()
    n=len(target);matrix=hand_jacobian.reshape(n,6,-1).transpose(1,2)
    a=matrix*weights[:,:,None];b=target*weights
    h=a.transpose(1,2)@a+(force_penalty+smooth_penalty)*torch.eye(6,device=a.device,dtype=a.dtype)
    rhs=(a.transpose(1,2)@b[:,:,None]).squeeze(-1)+smooth_penalty*previous_hands.reshape(n,6)
    hands=torch.linalg.solve(h,rhs[:,:,None]).reshape(n,2,3)
    norms=hands.norm(dim=-1)
    budget=(60-pelvis.norm(dim=-1)).clamp_min(0)
    scale=torch.minimum(30/norms.amax(-1).clamp_min(30),budget/norms.sum(-1).clamp_min(1e-9)).clamp_max(1)
    hands=hands*scale[:,None,None]
    residual=(torch.einsum('ndk,nk->nd',matrix,hands.flatten(1))-target)*weights
    return torch.cat((pelvis[:,None],hands),1),gate,yaw_target,residual.norm(dim=-1)
