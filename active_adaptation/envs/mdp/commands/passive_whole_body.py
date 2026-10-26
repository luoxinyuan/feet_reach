"""Continuous full-motion tracking with one to three upper-body contact forces."""
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp.commands.foot_reach import WallFootReachCommand
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
from active_adaptation.utils.body_guidance import allocate_forces,limit_force,slew_force,consistent_force


class PassiveWholeBodyCommand(MotionTrackingCommand):
    supports_symmetry_augmentation=False
    target_joint_pos_obs=WallFootReachCommand.target_joint_pos_obs
    target_joint_pos_obs_sym=WallFootReachCommand.target_joint_pos_obs_sym

    def __init__(self,env,*args,guidance_enabled=True,**kwargs):
        super().__init__(env,*args,**kwargs)
        self.guidance_enabled=guidance_enabled
        self.zero_init_prob=0.
        names=['torso_link']+[s+'_'+b for s in ['left','right'] for b in ['shoulder_roll_link','elbow_link','wrist_roll_link','hand_mimic']]
        self.point_ids=torch.tensor([self.asset.body_names.index(n) for n in names],device=self.device)
        self.motion_point_ids=torch.tensor([self.dataset.body_names.index(n) for n in names],device=self.device)
        n=self.num_envs
        self.slots=torch.zeros(n,3,dtype=torch.long,device=self.device)
        self.active=torch.ones(n,3,dtype=torch.bool,device=self.device)
        self.force=torch.zeros(n,3,3,device=self.device)
        self.desired=torch.zeros_like(self.force)
        self.remaining=torch.zeros(n,device=self.device)
        self.releasing=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.stiffness=torch.full((n,1,1),80.,device=self.device)
        self.rows=torch.arange(n,device=self.device)
        shape=(n,len(self.asset.body_names),3)
        self.force_apply_buffer=torch.zeros(shape,device=self.device)
        self.torque_apply_buffer=torch.zeros_like(self.force_apply_buffer)
        self.position_apply_buffer=torch.zeros_like(self.force_apply_buffer)
        self.force_apply_world=True
        self.log_step=0

    def update_reward_target(self):
        self.reward_keypoints_w=self._motion.body_pos_w[:,0]+self.env.scene.env_origins[:,None]
        self.reward_root_pos_w=self._motion.root_pos_w[:,0]+self.env.scene.env_origins
        self.reward_root_quat_w=self._motion.root_quat_w[:,0]

    def sample_init_robot(self,ids,motion,lift_height=0.):
        return super().sample_init_robot(ids,motion,lift_height=0.)

    def reset(self,ids):
        super().reset(ids)
        self._motion=self.dataset.get_slice(None,self.t,steps=self.future_steps)
        self.update_reward_target()
        self.force[ids]=0;self.desired[ids]=0
        self.force_apply_buffer[ids]=0;self.torque_apply_buffer[ids]=0
        self._reference()
        self._select(ids)
        self._solve()

    def _reference(self):
        nxt=self.dataset.get_slice(None,torch.minimum(self.t+1,self.lengths-1),1)
        self.anchor=nxt.body_pos_w[:,0,self.motion_point_ids]+self.env.scene.env_origins[:,None]
        self.ref_velocity=(nxt.body_pos_w[:,0,self.motion_point_ids]-self._motion.body_pos_w[:,0,self.motion_point_ids])/self.env.step_dt
        self.root_anchor=nxt.root_pos_w[:,0]+self.env.scene.env_origins
        self.ref_root_velocity=self._motion.root_lin_vel_w[:,0]
        err=yaw_from_quat(nxt.root_quat_w[:,0])-yaw_from_quat(self.asset.data.root_quat_w)
        self.yaw_error=torch.atan2(err.sin(),err.cos())

    def _intent(self):
        force=60.*(self.root_anchor-self.asset.data.root_pos_w)+8.*(self.ref_root_velocity-self.asset.data.root_lin_vel_w)
        force=force*(40./force.norm(dim=-1,keepdim=True).clamp_min(40.))
        yaw=(8.*self.yaw_error+1.5*(self._motion.root_ang_vel_w[:,0,2]-self.asset.data.root_ang_vel_w[:,2])).clamp(-4.,4.)
        return force,yaw

    def _local(self,slots):
        asset_ids=self.point_ids[slots]
        positions=self.asset.data.body_pos_w[self.rows[:,None],asset_ids]
        velocity=self.asset.data.body_lin_vel_w[self.rows[:,None],asset_ids]
        target=self.anchor[self.rows[:,None],slots]
        ref_vel=self.ref_velocity[self.rows[:,None],slots]
        return self.stiffness*(target-positions)+4.*(ref_vel-velocity),positions-self.asset.data.root_pos_w[:,None]

    def _select(self,ids):
        if not len(ids):return
        weights=.03+self.ref_velocity[ids].norm(dim=-1)
        self.slots[ids]=torch.multinomial(weights,3,replacement=False)
        counts=torch.randint(1,4,(len(ids),),device=self.device)
        self.active[ids]=torch.arange(3,device=self.device)[None]<counts[:,None]
        self.stiffness[ids]=torch.rand(len(ids),1,1,device=self.device)*40.+60.
        self.remaining[ids]=torch.rand(len(ids),device=self.device)*2.+1.
        self.releasing[ids]=False
        # Increase to three contacts when the sampled sparse layout poorly
        # expresses global intent. Soft residuals remain explicitly measured.
        local,arms=self._local(self.slots);fd,td=self._intent()
        proposed=limit_force(allocate_forces(local,arms,fd,td,self.active,torch.zeros_like(local)))
        ferr=(proposed.sum(1)-fd).norm(dim=-1)
        terr=(torch.cross(arms,proposed,dim=-1).sum(1)[:,2]-td).abs()
        bad=(ferr>5.+.5*fd.norm(dim=-1))|(terr>.5+.5*td.abs())
        expand=ids[bad[ids]]
        self.active[expand]=True

    def _solve(self):
        local,arms=self._local(self.slots);fd,td=self._intent()
        desired=allocate_forces(local,arms,fd,td,self.active,self.force)
        desired=consistent_force(desired,arms,fd,td,self.active)
        self.desired.copy_(limit_force(desired))
        self.desired[self.releasing]=0.
        if not self.guidance_enabled:self.desired.zero_()
        self.desired_force=fd;self.desired_yaw=td

    def before_update(self):
        super().before_update();self.update_reward_target();self._reference()
        self.remaining-=self.env.step_dt
        self.releasing|=self.remaining<=0
        ready=self.releasing&(self.force.norm(dim=-1).amax(-1)<1e-5)
        self._select(ready.nonzero().squeeze(-1));self._solve()
        self.log_step+=1
        if self.log_step%16==0:
            _,arms=self._local(self.slots)
            net=self.force.sum(1);yaw=torch.cross(arms,self.force,dim=-1).sum(1)[:,2]
            self.env.extra.update({
                'guidance/force_max_N':self.force.norm(dim=-1).max().item(),
                'guidance/force_budget_max_N':self.force.norm(dim=-1).sum(-1).max().item(),
                'guidance/contact_count':(self.force.norm(dim=-1)>1e-5).sum(-1).float().mean().item(),
                'guidance/net_residual_N':(net-self.desired_force).norm(dim=-1).mean().item(),
                'guidance/yaw_residual_Nm':(yaw-self.desired_yaw).abs().mean().item(),
                'guidance/yaw_opposed_fraction':((yaw*self.desired_yaw<-.001)&(self.desired_yaw.abs()>.1)).float().mean().item(),
                'guidance/net_opposed_fraction':((net*self.desired_force).sum(-1)<-.001).float().mean().item(),
                'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs().mean().item()})

    def step(self,substep):
        self.force.copy_(slew_force(self.force,self.desired,self.env.physics_dt))
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        ids=self.point_ids[self.slots]
        self.force_apply_buffer[self.rows[:,None],ids]=self.force
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.asset.has_external_wrench=False

    @observation
    def upper_body_forces_b(self):
        forces=self.force_apply_buffer[:,self.point_ids]
        q=self.asset.data.root_quat_w[:,None].expand(-1,len(self.point_ids),-1)
        return quat_apply_inverse(q,forces).flatten(1)

    def upper_body_forces_b_sym(self):
        return SymmetryTransform(torch.arange(27),torch.ones(27))
