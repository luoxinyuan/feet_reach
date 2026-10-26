"""Standing, target-private EE compliance with continuous local transitions."""
import torch
from active_adaptation.envs.mdp import observation
from active_adaptation.envs.mdp.commands.foot_reach import WallFootReachCommand
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.passive_compliance import fit_spring, spring_force


class PassiveComplianceCommand(WallFootReachCommand):
    def __init__(self, env, *args, hold_probability=.2, transition_timeout=100,
                 force_limit=60., damping=8., **kwargs):
        super().__init__(env, *args, **kwargs)
        self.hand_ids = [self.asset.body_names.index(s+'_hand_mimic') for s in ('left','right')]
        self.motion_hand_ids = [self.dataset.body_names.index(s+'_hand_mimic') for s in ('left','right')]
        self.hold_probability = hold_probability
        self.transition_timeout = transition_timeout
        self.force_limit = force_limit
        self.damping = damping
        self.distance_max = .02
        self.anchor = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.stiffness = torch.full((self.num_envs,2),80.,device=self.device)
        self.active_hands = torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self.ee_force = torch.zeros_like(self.anchor)
        self.age = torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.dwell = torch.zeros_like(self.age)
        self.force_apply_world = True
        self.transition_count = 0

    def _refresh(self):
        self._motion = self.dataset.get_slice(None, self.t, steps=torch.zeros_like(self.future_steps))
        self._motion.joint_vel.zero_()
        self._motion.root_lin_vel_w.zero_()
        self._motion.root_ang_vel_w.zero_()
        self.feet_standing[:] = True
        self.update_reward_target()

    def sample_init(self, env_ids):
        super().sample_init(env_ids)
        self._choose_goal(env_ids)

    def _choose_goal(self, ids):
        if len(ids)==0: return
        start = self.t[ids].long().clone()
        # Same-clip neighbors preserve a consistent IK branch; both directions.
        offsets = torch.arange(-100,101,5,device=self.device)
        candidates = (start[:,None] + offsets).clamp_min(0)
        candidates = torch.minimum(candidates,self.lengths[ids,None]-1)
        # get_slice takes common relative steps; arbitrary per-env starts are
        # obtained by flattening through repeated env ids.
        repeated = ids[:,None].expand_as(candidates).reshape(-1)
        poses = self.dataset.get_slice(repeated,candidates.reshape(-1),1).body_pos_w[:,0,self.motion_hand_ids]
        poses = poses.reshape(len(ids),len(offsets),2,3)
        initial = self.dataset.get_slice(ids,start,1).body_pos_w[:,0,self.motion_hand_ids]
        distance = (poses-initial[:,None]).norm(dim=-1).amax(-1)
        radius = torch.empty(len(ids),1,device=self.device).uniform_(.02,max(.02001,self.distance_max))
        valid = (distance>=.01)&(distance<=radius)
        weights = valid.float()
        no_neighbor = ~valid.any(-1)
        weights[no_neighbor,len(offsets)//2]=1.
        selected = torch.multinomial(weights,1).squeeze(-1)
        goal = candidates[torch.arange(len(ids),device=self.device),selected]
        hold = (torch.rand(len(ids),device=self.device)<self.hold_probability)|no_neighbor
        goal[hold]=start[hold]
        # Fit the scalar stiffness independently for each hand along the
        # selected reference interval, with fixed damping and bounded gain.
        alpha = torch.linspace(0,1,12,device=self.device)
        times = (start[:,None]+(goal-start)[:,None]*alpha).round().long()
        path = self.dataset.get_slice(ids[:,None].expand_as(times).reshape(-1),times.reshape(-1),1).body_pos_w[:,0,self.motion_hand_ids]
        path = path.reshape(len(ids),12,2,3)
        duration = ((goal-start).abs().float()*.02).clamp_min(.2)
        self.stiffness[ids] = fit_spring(path,duration[:,None,None,None]/11.,self.damping)
        self.anchor[ids] = path[:,-1]+self.env.scene.env_origins[ids,None]
        self.active_hands[ids] = ((path[:,-1]-initial).norm(dim=-1)>.002)&~hold[:,None]
        self.t[ids] = goal.to(self.t.dtype)
        self.age[ids]=0
        self.dwell[ids]=0
        self.transition_count += len(ids)
        self._refresh()

    def reset(self, env_ids):
        super().reset(env_ids)
        self.ee_force[env_ids]=0.
        self.age[env_ids]=0
        self.dwell[env_ids]=0
        self.env.episode_length_buf[env_ids]=0
        self._refresh()

    def before_update(self):
        self.finished[:] = False
        self.boot_indicator.sub_(1).clamp_min_(0)
        self._refresh()
        self.age += 1
        joint_error = (self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs().mean(-1)
        ee_error = (self.asset.data.body_pos_w[:,self.hand_ids]-self.anchor).norm(dim=-1).amax(-1)
        settled = (joint_error<.08)&(ee_error<.02)&(self.asset.data.joint_vel.abs().mean(-1)<.2)
        self.dwell = torch.where(settled,self.dwell+1,torch.zeros_like(self.dwell))
        complete = (self.dwell>=10)&(self.age>=25)
        timed_out = self.age>=self.transition_timeout
        if hasattr(self.env,'extra'):
            self.env.extra.update({'compliance/joint_mae_rad':joint_error.mean().item(),
                'compliance/ee_error_m':ee_error.mean().item(),
                'compliance/force_mean_N':self.ee_force.norm(dim=-1).mean().item(),
                'compliance/force_max_N':self.ee_force.norm(dim=-1).max().item(),
                'compliance/success_fraction':complete.float().mean().item(),
                'compliance/timeout_fraction':timed_out.float().mean().item(),
                'compliance/distance_max_m':self.distance_max})
        self._choose_goal((complete|timed_out).nonzero().squeeze(-1))

    def step_schedule(self, progress):
        self.distance_max = .02+.06*min(1.,progress*2)

    def step(self, substep):
        self.ee_force.copy_(spring_force(self.anchor,self.asset.data.body_pos_w[:,self.hand_ids],
            self.asset.data.body_lin_vel_w[:,self.hand_ids],self.stiffness,self.active_hands,
            self.damping,self.force_limit))
        self.force_apply_buffer.zero_()
        self.force_apply_buffer[:,self.hand_ids] = self.ee_force
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.asset.has_external_wrench=False

    @observation
    def ee_external_force_b(self):
        quat = self.asset.data.root_quat_w[:,None].expand(-1,2,-1)
        return quat_apply_inverse(quat,self.ee_force).flatten(1)

    def ee_external_force_b_sym(self):
        return SymmetryTransform(torch.arange(6),torch.ones(6))
