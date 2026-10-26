"""Continuous AMASS tracking; future EE anchors never advance reference time."""
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.foot_reach import WallFootReachCommand
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.passive_compliance import fit_spring, spring_force


class PassiveEEContinuousCommand(WallFootReachCommand):
    def __init__(self, env, *args, transition_steps=(25,50), force_limit=30.,
                 damping=8., ramp_seconds=.2, guidance_enabled=True,
                 guidance_mode='transition', spring_stiffness=80., **kwargs):
        if guidance_mode not in ('transition', 'next_frame'):
            raise ValueError(f'Unknown guidance_mode: {guidance_mode}')
        if spring_stiffness <= 0:
            raise ValueError('spring_stiffness must be positive')
        self.guidance_mode=guidance_mode;self.spring_stiffness=spring_stiffness
        super().__init__(env,*args,**kwargs)
        self.hand_ids=[self.asset.body_names.index(s+'_hand_mimic') for s in ('left','right')]
        self.motion_hand_ids=[self.dataset.body_names.index(s+'_hand_mimic') for s in ('left','right')]
        self.transition_steps=transition_steps;self.force_limit=force_limit;self.damping=damping
        self.ramp_seconds=ramp_seconds;self.guidance_enabled=guidance_enabled
        self.anchor=torch.zeros(self.num_envs,2,3,device=self.device)
        self.ee_force=torch.zeros_like(self.anchor);self.start_force=torch.zeros_like(self.anchor)
        self.stiffness=torch.full((self.num_envs,2),80.,device=self.device)
        self.active_hands=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self.goal_end=torch.zeros_like(self.t);self.ramp_elapsed=torch.zeros(self.num_envs,device=self.device)
        self.hold=self.dataset.get_slice_info(torch.arange(self.num_envs,device=self.device))['is_hold'][:,0]>.5
        self.force_apply_world=True;self.transition_count=0;self.log_step=0

    def _choose_goal(self,ids):
        if self.guidance_mode=='next_frame':return
        if len(ids)==0:return
        span=torch.randint(*[self.transition_steps[0],self.transition_steps[1]+1],(len(ids),),device=self.device)
        self.goal_end[ids]=torch.minimum(self.t[ids].long()+span,self.lengths[ids]-1).to(self.t.dtype)
        # Fit on the source interval, without changing self.t or the reward reference.
        alpha=torch.linspace(0,1,12,device=self.device)
        times=(self.t[ids,None]+(self.goal_end[ids]-self.t[ids])[:,None]*alpha).round().long()
        path=self.dataset.get_slice(ids[:,None].expand_as(times).reshape(-1),times.reshape(-1),1).body_pos_w[:,0,self.motion_hand_ids].reshape(len(ids),12,2,3)
        duration=((self.goal_end[ids]-self.t[ids]).float()*self.env.step_dt).clamp_min(.02)
        self.stiffness[ids]=fit_spring(path,duration[:,None,None,None]/11.,self.damping)
        self.anchor[ids]=path[:,-1]+self.env.scene.env_origins[ids,None]
        self.active_hands[ids]=((path[:,-1]-path[:,0]).norm(dim=-1)>.002)&~self.hold[ids,None]
        self.start_force[ids]=self.ee_force[ids];self.ramp_elapsed[ids]=0
        self.transition_count+=len(ids)

    def sample_init(self,ids):
        super().sample_init(ids)
        self.hold[ids]=self.dataset.get_slice_info(ids)['is_hold'][:,0]>.5
        self.ee_force[ids]=0.;self._choose_goal(ids)

    def reset(self,ids):
        super().reset(ids)
        self.env.episode_length_buf[ids]=0
        if self.guidance_mode=='next_frame':self._update_next_frame_anchor()

    def _update_next_frame_anchor(self):
        # Explicit t+1: _motion's future slots are [0,2,4,8,16], not consecutive.
        times=torch.minimum(self.t+1,self.lengths-1)
        target=self.dataset.get_slice(None,times,1)
        self.anchor.copy_(target.body_pos_w[:,0,self.motion_hand_ids]+self.env.scene.env_origins[:,None])

    def before_update(self):
        # Parent advances by exactly one frame, retains source velocities,
        # and uses the fixed world-space reference for reward computation.
        super().before_update()
        if self.guidance_mode=='next_frame':self._update_next_frame_anchor()
        self._choose_goal(((self.t>=self.goal_end)&~self.finished).nonzero().squeeze(-1))
        self.log_step+=1
        if self.log_step%16==0:
            target=self._motion.body_pos_w[:,0,self.motion_hand_ids]+self.env.scene.env_origins[:,None]
            self.env.extra.update({'tracking/ee_error_m':(self.asset.data.body_pos_w[:,self.hand_ids]-target).norm(dim=-1).mean().item(),
             'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs().mean().item(),
             'guidance/force_max_N':self.ee_force.norm(dim=-1).max().item(),
             'guidance/hold_fraction':self.hold.float().mean().item()})

    def step(self,substep):
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        if self.guidance_mode=='next_frame':
            self.ee_force.copy_((self.anchor-self.asset.data.body_pos_w[:,self.hand_ids])*self.spring_stiffness)
            self.ee_force.mul_(self.force_limit/self.ee_force.norm(dim=-1,keepdim=True).clamp_min(self.force_limit))
            self.ee_force.mul_((~self.hold)[:,None,None])
            if not self.guidance_enabled:self.ee_force.zero_()
        elif self.guidance_enabled:
            raw=spring_force(self.anchor,self.asset.data.body_pos_w[:,self.hand_ids],
                self.asset.data.body_lin_vel_w[:,self.hand_ids],self.stiffness,self.active_hands,self.damping,self.force_limit)
            self.ramp_elapsed.add_(self.env.physics_dt)
            alpha=(self.ramp_elapsed/self.ramp_seconds).clamp(0,1)[:,None,None]
            self.ee_force.copy_(self.start_force*(1-alpha)+raw*alpha)
        else:self.ee_force.zero_()
        self.force_apply_buffer[:,self.hand_ids]=self.ee_force
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w);self.asset.has_external_wrench=False

    @observation
    def ee_external_force_b(self):
        return quat_apply_inverse(self.asset.data.root_quat_w[:,None].expand(-1,2,-1),self.ee_force).flatten(1)

    def ee_external_force_b_sym(self):
        return SymmetryTransform(torch.arange(6),torch.ones(6))
