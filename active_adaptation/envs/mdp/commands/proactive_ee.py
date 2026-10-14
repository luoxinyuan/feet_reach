"""EE tracking with independently randomized wall and metric surface point clouds.
Geometry is a noisy synthetic front-surface cloud, NOT a rendered depth sensor.
No distance-triggered braking or target projection is used.
"""
import torch
from active_adaptation.envs.mdp import observation, reward
from active_adaptation.envs.mdp.commands.foot_reach import WallFootReachCommand
from active_adaptation.envs.mdp.commands.motion_tracking import _calc_exp_sigma
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform

class ProactiveEECommand(WallFootReachCommand):
    def __init__(self, env, *args, geometry_points=128, no_wall_prob=.25,
                 wall_range=(.40,.50), force_soft=20., force_rate_soft=500.,
                 contact_threshold_N=1., contact_target_margin_m=.005,
                 contact_grace_steps=8, contact_miss_ramp_steps=8,
                 contact_late_penalty_max=.5,
                 unexpected_contact_score=3.0, **kwargs):
        super().__init__(env,*args,**kwargs)
        self.zero_init_prob=1.  # Always approach from a collision-free start.
        self.npoints=geometry_points
        self.no_wall_prob=no_wall_prob
        self.wall_range=wall_range
        self.force_soft=force_soft;self.force_rate_soft=force_rate_soft
        self.contact_threshold_N=float(contact_threshold_N)
        self.contact_target_margin_m=float(contact_target_margin_m)
        self.contact_grace_steps=int(contact_grace_steps)
        self.contact_miss_ramp_steps=int(contact_miss_ramp_steps)
        self.contact_late_penalty_max=float(contact_late_penalty_max)
        self.unexpected_contact_score=float(unexpected_contact_score)
        if self.contact_grace_steps < 0 or self.contact_miss_ramp_steps < 1:
            raise ValueError('contact_grace_steps must be nonnegative and contact_miss_ramp_steps positive')
        if self.contact_late_penalty_max < 0.:
            raise ValueError('contact_late_penalty_max must be nonnegative')
        self.wall=env.scene['reach_wall']
        self.sensor=env.scene['ee_contact_forces']
        self.hand_sensor=self.sensor.body_names.index('left_hand_mimic')
        self.wall_sensor=env.scene['ee_wall_contact_forces']
        self.wall_contact_this_step=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.contact_required_age=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.wall_x=torch.full((self.num_envs,),.45,device=self.device)
        self.wall_present=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.previous_force=torch.zeros(self.num_envs,device=self.device)
        self.peak_force=torch.zeros_like(self.previous_force)
        self.peak_rate=torch.zeros_like(self.previous_force)
        self.cloud=torch.zeros(self.num_envs,self.npoints,4,device=self.device)
        self.cloud_world=torch.zeros(self.num_envs,self.npoints,3,device=self.device)
        self.cloud_valid=torch.zeros(self.num_envs,self.npoints,device=self.device)
        self.cloud_counter=0
        self._external_ee_target=None
        self.left_foot_motion=self.dataset.body_names.index('left_ankle_roll_link')
        self.right_foot_motion=self.dataset.body_names.index('right_ankle_roll_link')
        self.left_foot_asset=self.asset.body_names.index('left_ankle_roll_link')
        self.right_foot_asset=self.asset.body_names.index('right_ankle_roll_link')

    def reset(self,env_ids):
        super().reset(env_ids)
        n=len(env_ids)
        self.wall_present[env_ids]=torch.rand(n,device=self.device)>self.no_wall_prob
        self.wall_x[env_ids]=torch.empty(n,device=self.device).uniform_(*self.wall_range)
        state=self.wall.data.default_root_state[env_ids].clone()
        state[:,:3]=self.env.scene.env_origins[env_ids]
        state[:,0]+=torch.where(self.wall_present[env_ids],self.wall_x[env_ids]+.025,torch.full((n,),3.,device=self.device))
        state[:,1]+=.15;state[:,2]+=1.
        state[:,7:]=0
        self.wall.write_root_state_to_sim(state,env_ids=env_ids)
        self.previous_force[env_ids]=0;self.peak_force[env_ids]=0;self.peak_rate[env_ids]=0
        self.wall_contact_this_step[env_ids]=False
        self.contact_required_age[env_ids]=0
        self.refresh_cloud(env_ids)

    def step(self,substep):
        if substep==0:
            self.peak_force.zero_();self.peak_rate.zero_()
            self.wall_contact_this_step.zero_()

    def post_step(self,substep):
        # Read each physics substep, not only the 50 Hz control boundary.
        force=(-self.sensor.data.net_forces_w[:,self.hand_sensor,0]).clamp_min(0)
        rate=((force-self.previous_force)/self.env.physics_dt).clamp_min(0)
        self.peak_force.copy_(torch.maximum(self.peak_force,force))
        self.peak_rate.copy_(torch.maximum(self.peak_rate,rate))
        self.previous_force.copy_(force)
        # Latch wall contact across physics substeps so even a brief contact
        # counts in the control-step reward. The dedicated sensor is filtered
        # to the wall, so unrelated hand contacts cannot satisfy this term.
        matrix=self.wall_sensor.data.force_matrix_w
        if matrix is not None:
            wall_force=matrix.reshape(self.num_envs,-1,3).sum(dim=1)
            wall_normal_force=(-wall_force[:,0]).clamp_min(0.)
            self.wall_contact_this_step |= wall_normal_force>self.contact_threshold_N

    def contact_requirement_state(self):
        target=self.target_ee_world()
        wall_front=self.wall_x+self.env.scene.env_origins[:,0]
        required=self.wall_present & (target[:,0]>wall_front+self.contact_target_margin_m)
        return required,self.wall_contact_this_step

    @reward
    def contact_requirement(self):
        """Allow time for approach, then ramp a cost until required contact occurs.

        Returned values are already in weighted reward units. The task config uses
        weight 1, so the environment's usual control-dt scaling applies directly.
        """
        required,contact=self.contact_requirement_state()
        self.contact_required_age=torch.where(
            required,self.contact_required_age+1,torch.zeros_like(self.contact_required_age))
        late_steps=(self.contact_required_age-self.contact_grace_steps).clamp_min(0)
        ramp=(late_steps.float()/self.contact_miss_ramp_steps).clamp(max=1.)
        score=torch.zeros(self.num_envs,device=self.device)
        score[required & ~contact]=-self.contact_late_penalty_max*ramp[required & ~contact]
        score[~required & contact]=-self.unexpected_contact_score
        return score.unsqueeze(-1)

    def before_update(self):
        super().before_update()
        if hasattr(self,'cloud_counter'):
            self.cloud_counter+=1
            if self.cloud_counter%16==0 and hasattr(self.env,'extra'):
                required, contact = self.contact_requirement_state()
                required_count=required.sum().clamp_min(1)
                optional_count=(~required).sum().clamp_min(1)
                distance=self.wall_x+self.env.scene.env_origins[:,0]-self.asset.data.body_pos_w[:,self.hand_asset,0]
                self.env.extra.update({
                    'contact/mean_peak_N': self.peak_force.mean().item(),
                    'contact/max_peak_N': self.peak_force.max().item(),
                    'contact/max_loading_rate_N_s': self.peak_rate.max().item(),
                    'contact/fraction': (self.peak_force>1.).float().mean().item(),
                    'contact/required_fraction': required.float().mean().item(),
                    'contact/requirement_match_fraction': (required==contact).float().mean().item(),
                    'contact/required_hit_fraction': (required & contact).sum().float().div(required_count).item(),
                    'contact/required_miss_fraction': (required & ~contact).sum().float().div(required_count).item(),
                    'contact/late_miss_fraction': (required & ~contact & (self.contact_required_age>self.contact_grace_steps)).sum().float().div(required_count).item(),
                    'contact/required_age_steps': self.contact_required_age[required].float().mean().item() if required.any() else 0.,
                    'contact/unexpected_fraction': ((~required) & contact).sum().float().div(optional_count).item(),
                    'tracking/ee_error_m': (self.target_ee_world()-self.asset.data.body_pos_w[:,self.hand_asset]).norm(dim=-1).mean().item(),
                    'geometry/wall_fraction': self.wall_present.float().mean().item(),
                    'geometry/mean_hand_wall_gap_m': distance[self.wall_present].mean().item() if self.wall_present.any() else 0.,
                })
            if self.cloud_counter%2==0:
                self.refresh_cloud(torch.arange(self.num_envs,device=self.device))

    def refresh_cloud(self,env_ids):
        n=len(env_ids)
        p=torch.rand(n,self.npoints,3,device=self.device)
        p[:,:,0]=self.wall_x[env_ids,None]
        p[:,:,1]=p[:,:,1]*1.0-.35
        p[:,:,2]=p[:,:,2]+.5
        p+=torch.randn_like(p)*.003
        p+=self.env.scene.env_origins[env_ids,None]
        self.cloud_world[env_ids]=p
        self.cloud_valid[env_ids]=((torch.rand(n,self.npoints,device=self.device)>.1)&self.wall_present[env_ids,None]).float()

    def set_target_ee_pos_b(self,target):
        if target is None:self._external_ee_target=None;return
        target=torch.as_tensor(target,device=self.device,dtype=torch.float32)
        if target.shape==(3,):target=target.expand(self.num_envs,-1)
        if target.shape!=(self.num_envs,3) or not torch.isfinite(target).all():raise ValueError('Expected finite root-frame xyz')
        self._external_ee_target=target.clone()

    def target_ee_world(self):
        if self._external_ee_target is not None:
            from active_adaptation.utils.math import quat_apply
            return self.asset.data.root_pos_w+quat_apply(self.asset.data.root_quat_w,self._external_ee_target)
        return self._motion.body_pos_w[:,0,self.hand_motion]+self.env.scene.env_origins

    @observation
    def target_ee_pos_b(self):
        return quat_apply_inverse(self.asset.data.root_quat_w,self.target_ee_world()-self.asset.data.root_pos_w)

    def target_ee_pos_b_sym(self):
        return SymmetryTransform(torch.arange(3),[1.,1.,1.])

    @observation
    def geometry_cloud(self):
        relative=self.cloud_world-self.asset.data.root_pos_w[:,None]
        quat=self.asset.data.root_quat_w[:,None].expand(-1,self.npoints,-1)
        self.cloud[:,:,:3]=quat_apply_inverse(quat,relative)
        self.cloud[:,:,3]=self.cloud_valid
        self.cloud[:,:,:3]*=self.cloud_valid[:,:,None]
        return self.cloud.flatten(1).clone()

    def geometry_cloud_sym(self):
        return SymmetryTransform(torch.arange(self.npoints*4),torch.ones(self.npoints*4))

    @reward
    def left_ee_tracking(self):
        """Track only the commanded hand, using the original low-level EE kernel.

        The target is the same root-frame xyz exposed to the student, including
        external evaluation/teleoperation commands. The right arm is constrained
        by full-body joint tracking, not an additional Cartesian task.
        """
        actual_b = quat_apply_inverse(
            self.asset.data.root_quat_w,
            self.asset.data.body_pos_w[:, self.hand_asset] - self.asset.data.root_pos_w,
        )
        error = (self.target_ee_pos_b() - actual_b).norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["ee"])

    # Legacy task rewards remain callable for historical checkpoint configs.
    @reward
    def reach_tracking(self):
        error=(self.target_ee_world()-self.asset.data.body_pos_w[:,self.hand_asset]).square().sum(-1,keepdim=True)
        # Same objective with and without walls. No contact-conditioned switch.
        return torch.exp(-error/.01)

    @reward
    def double_support_tracking(self):
        ids=[self.left_foot_asset,self.right_foot_asset];mi=[self.left_foot_motion,self.right_foot_motion]
        target=self._motion.body_pos_w[:,0,mi]+self.env.scene.env_origins[:,None]
        error=(target-self.asset.data.body_pos_w[:,ids]).norm(dim=-1).mean(-1,keepdim=True)
        return torch.exp(-error/.025)

    @reward
    def contact_force_cost(self):
        # Legacy checkpoint compatibility; omitted from the current task reward.
        return -((self.peak_force-self.force_soft).clamp_min(0)/self.force_soft).square().unsqueeze(-1)

    @reward
    def contact_jump_cost(self):
        return -((self.peak_rate-self.force_rate_soft).clamp_min(0)/self.force_rate_soft).square().unsqueeze(-1)

    def debug_draw(self):
        if hasattr(self.env,'debug_draw'):
            self.env.debug_draw.point(self.target_ee_world(),color=(1.,.1,.1,1.),size=12.)
