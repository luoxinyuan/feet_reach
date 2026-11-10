"""Pelvis reference-delta spring, two-hand joint/yaw solver, causal force history."""
import torch
from .online_estimated_compliance import OnlineEstimatedComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
from active_adaptation.utils.independent_pelvis_guidance import allocate_independent

class IndependentPelvisComplianceCommand(OnlineEstimatedComplianceCommand):
    def __init__(self,env,*args,pelvis_gain=1190.6272466177975,yaw_delta_gain=83.,
                 yaw_gate_low=.05,yaw_gate_high=.3,**kwargs):
        super().__init__(env,*args,**kwargs)
        if self.point_names!=['pelvis','left_hand_mimic','right_hand_mimic']:
            raise ValueError('Independent pelvis requires canonical root3 point order')
        self.pelvis_gain=float(pelvis_gain);self.yaw_delta_gain=float(yaw_delta_gain)
        self.yaw_gate_low=float(yaw_gate_low);self.yaw_gate_high=float(yaw_gate_high)
        if not 0<=self.yaw_gate_low<self.yaw_gate_high:raise ValueError('Invalid yaw gate')
        upper=[i for i,name in enumerate(self.asset.joint_names) if any(k in name for k in ('waist','shoulder','elbow','wrist'))]
        self.upper_joint_ids=torch.tensor(upper,device=self.device)
        self.fit_columns=torch.tensor([5]+[6+i for i in upper],device=self.device)
        self.slots[:]=torch.arange(3,device=self.device);self.active[:]=True

    def _reference(self):
        # Reuse actual-posture point Jacobians and unchanged joint delta+feedback target.
        super()._reference()
        nxt=self.dataset.get_slice(None,torch.minimum(self.t+1,self.lengths-1),1)
        self.root_delta=nxt.root_pos_w[:,0]-self._motion.root_pos_w[:,0]
        angle=yaw_from_quat(nxt.root_quat_w[:,0])-yaw_from_quat(self._motion.root_quat_w[:,0])
        self.reference_yaw_delta=torch.atan2(angle.sin(),angle.cos())

    def _select(self,ids):
        # Fixed semantic channels: never randomly drop pelvis or discard reference segments.
        self.slots[ids]=torch.arange(3,device=self.device);self.active[ids]=True
        self.releasing[ids]=False;self.remaining[ids]=0

    def _solve(self):
        force,gate,yaw,error=allocate_independent(self.root_delta,self.reference_yaw_delta,
            self.point_jacobians[:,1:,:,self.fit_columns],
            self.generalized_target[:,6+self.upper_joint_ids],self.force[:,1:],
            dt=self.env.step_dt,pelvis_gain=self.pelvis_gain,yaw_gain=self.yaw_delta_gain,
            yaw_gate_low=self.yaw_gate_low,yaw_gate_high=self.yaw_gate_high,
            force_penalty=self.force_penalty,smooth_penalty=self.smooth_penalty)
        self.desired.copy_(force)
        if not self.guidance_enabled:self.desired.zero_()
        self.desired_force=force[:,0];self.desired_yaw=yaw
        self.yaw_gate=gate;self.hand_fit_error=error

    def before_update(self):
        # No inherited random release/reselection or old root-wrench projections.
        MotionTrackingCommand.before_update(self)
        self.update_reward_target();self._reference();self._solve();self.log_step+=1
        if self.log_step%16==0:
            self.env.extra.update({
                'guidance/pelvis_mean_N':self.force[:,0].norm(dim=-1).mean().item(),
                'guidance/pelvis_active_fraction':(self.force[:,0].norm(dim=-1)>.05).float().mean().item(),
                'guidance/yaw_gate_mean':self.yaw_gate.mean().item(),
                'guidance/hand_fit_residual':self.hand_fit_error.mean().item(),
                'guidance/force_max_N':self.force.norm(dim=-1).max().item(),
                'guidance/force_budget_max_N':self.force.norm(dim=-1).sum(-1).max().item(),
                'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs().mean().item()})
