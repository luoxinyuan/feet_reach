"""AMASS playback with private transition anchors and simple waist/torso guidance."""
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform,joint_space_symmetry
from active_adaptation.utils.simple_locomotion_guidance import guidance,yaw_from_quat


class PassiveLocomotionCommand(MotionTrackingCommand):
    supports_symmetry_augmentation=False

    def __init__(self,env,*args,transition_steps=(25,50),ramp_seconds=.2,
                 stiffness=60.,force_limit=30.,yaw_stiffness=8.,torque_limit=4.,**kwargs):
        super().__init__(env,*args,**kwargs)
        self.zero_init_prob=0.
        self.transition_steps=transition_steps
        self.ramp_seconds=ramp_seconds
        self.stiffness=stiffness;self.force_limit=force_limit
        self.yaw_stiffness=yaw_stiffness;self.torque_limit=torque_limit
        self.waist_id=self.asset.body_names.index('pelvis')
        self.torso_id=self.asset.body_names.index('torso_link')
        self.anchor=torch.zeros(self.num_envs,3,device=self.device)
        self.target_yaw=torch.zeros(self.num_envs,device=self.device)
        self.goal_end=torch.zeros_like(self.t)
        self.hold=self.dataset.get_slice_info(torch.arange(self.num_envs,device=self.device))['is_hold'][:,0]>.5
        self.applied_force=torch.zeros_like(self.anchor);self.applied_torque=torch.zeros_like(self.anchor)
        self.start_force=torch.zeros_like(self.anchor);self.start_torque=torch.zeros_like(self.anchor)
        self.ramp_elapsed=torch.zeros(self.num_envs,device=self.device)
        shape=(self.num_envs,len(self.asset.body_names),3)
        self.force_apply_buffer=torch.zeros(shape,device=self.device)
        self.torque_apply_buffer=torch.zeros_like(self.force_apply_buffer)
        self.position_apply_buffer=torch.zeros_like(self.force_apply_buffer)
        self.force_apply_world=True
        self.transition_count=0
        self.log_step=0

    def _choose_goal(self,ids):
        if not len(ids):return
        span=torch.randint(self.transition_steps[0],self.transition_steps[1]+1,(len(ids),),device=self.device)
        self.goal_end[ids]=torch.minimum(self.t[ids].long()+span,self.lengths[ids]-1).to(self.t.dtype)
        target=self.dataset.get_slice(ids,self.goal_end[ids],1)
        self.anchor[ids]=target.root_pos_w[:,0]+self.env.scene.env_origins[ids]
        self.target_yaw[ids]=yaw_from_quat(target.root_quat_w[:,0])
        self.start_force[ids]=self.applied_force[ids]
        self.start_torque[ids]=self.applied_torque[ids]
        self.ramp_elapsed[ids]=0.
        self.transition_count+=len(ids)

    def sample_init(self,ids):
        super().sample_init(ids)
        self.hold[ids]=self.dataset.get_slice_info(ids)['is_hold'][:,0]>.5
        self.applied_force[ids]=0.;self.applied_torque[ids]=0.
        self._choose_goal(ids)

    def sample_init_robot(self,ids,motion,lift_height=0.):
        return super().sample_init_robot(ids,motion,lift_height=0.)

    def reset(self,ids):
        super().reset(ids)
        self.force_apply_buffer[ids]=0.;self.torque_apply_buffer[ids]=0.
        self.env.episode_length_buf[ids]=0
        self._motion=self.dataset.get_slice(None,self.t,steps=self.future_steps)
        self.update_reward_target()

    def update_reward_target(self):
        self.reward_keypoints_w=self._motion.body_pos_w[:,0]+self.env.scene.env_origins[:,None]
        self.reward_root_pos_w=self._motion.root_pos_w[:,0]+self.env.scene.env_origins
        self.reward_root_quat_w=self._motion.root_quat_w[:,0]

    def before_update(self):
        # Playback preserves the intermediate gait; only the private force
        # anchor jumps to a future frame at transition boundaries.
        super().before_update()
        self.update_reward_target()
        self._choose_goal(((self.t>=self.goal_end)&~self.finished).nonzero().squeeze(-1))
        self.log_step+=1
        if self.log_step%16==0 and hasattr(self.env,'extra'):
            joint=(self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs().mean()
            self.env.extra.update({'guidance/force_mean_N':self.applied_force.norm(dim=-1).mean().item(),
                'guidance/force_max_N':self.applied_force.norm(dim=-1).max().item(),
                'guidance/yaw_torque_max_Nm':self.applied_torque[:,2].abs().max().item(),
                'guidance/hold_fraction':self.hold.float().mean().item(),
                'tracking/joint_mae_rad':joint.item(),
                'tracking/root_xy_error_m':(self.asset.data.root_pos_w[:,:2]-self.reward_root_pos_w[:,:2]).norm(dim=-1).mean().item()})

    def step(self,substep):
        raw_force,raw_torque=guidance(self.asset.data.root_pos_w,self.asset.data.root_quat_w,
            self.anchor,self.target_yaw,self.hold,self.stiffness,self.force_limit,self.yaw_stiffness,self.torque_limit)
        self.ramp_elapsed.add_(self.env.physics_dt)
        alpha=(self.ramp_elapsed/self.ramp_seconds).clamp(0.,1.)[:,None]
        self.applied_force.copy_(self.start_force*(1-alpha)+raw_force*alpha)
        self.applied_torque.copy_(self.start_torque*(1-alpha)+raw_torque*alpha)
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        self.force_apply_buffer[:,self.waist_id]=self.applied_force
        self.torque_apply_buffer[:,self.torso_id]=self.applied_torque
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.asset.has_external_wrench=False

    @observation
    def body_guidance_wrench_b(self):
        q=self.asset.data.root_quat_w
        return torch.cat((quat_apply_inverse(q,self.applied_force),quat_apply_inverse(q,self.applied_torque)),-1)

    def body_guidance_wrench_b_sym(self):
        return SymmetryTransform(torch.arange(6),torch.ones(6))

    @observation
    def target_joint_pos_obs(self):
        return self._motion.joint_pos[:,0,self.joint_idx_motion]

    def target_joint_pos_obs_sym(self):
        return joint_space_symmetry(self.asset,[self.dataset.joint_names[i] for i in self.joint_idx_motion.tolist()])
