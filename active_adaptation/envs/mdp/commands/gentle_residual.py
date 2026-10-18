"""Matched A/B reward experiments over frozen CEER and the same PointNet actor."""
import torch
from active_adaptation.envs.mdp import reward
from active_adaptation.envs.mdp.commands.proactive_residual import ProactiveResidualCommand
from active_adaptation.envs.mdp.commands.motion_tracking import _calc_exp_sigma
from active_adaptation.utils.gentle_contact import ContactEvents


class GentleResidualCommand(ProactiveResidualCommand):
    def __init__(self, env, *args, gentle=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.gentle = dict(gentle or {})
        self.curriculum_enabled = self.gentle.get('curriculum_enabled', True)
        self.gentle_factor = 0. if self.curriculum_enabled else 1.
        self.events = ContactEvents(self.num_envs, self.device, env.physics_dt)
        self.event_impulse_step = torch.zeros(self.num_envs, device=self.device)
        self.event_peak_step = torch.zeros_like(self.event_impulse_step)
        self.event_growth_step = torch.zeros_like(self.event_impulse_step)
        self.contact_all_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_paid = torch.zeros_like(self.contact_all_step)

    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, 'events'):
            self.events.reset(env_ids)
            for x in (self.event_impulse_step, self.event_peak_step, self.event_growth_step,
                      self.contact_all_step, self.hold_steps, self.success_paid):
                x[env_ids] = 0

    def step_schedule(self, progress, iter=0):
        start = self.gentle.get('ramp_start', .05)
        end = self.gentle.get('ramp_end', .20)
        self.gentle_factor = min(max((progress-start)/(end-start), 0.), 1.) if self.curriculum_enabled else 1.

    def step(self, substep):
        super().step(substep)
        if substep == 0:
            self.event_impulse_step.zero_()
            self.event_peak_step.zero_()
            self.event_growth_step.zero_()
            self.contact_all_step.fill_(True)

    def post_step(self, substep):
        super().post_step(substep)
        matrix = self.wall_sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError('Gentle reward requires wall-filtered contact forces')
        force = (-matrix.reshape(self.num_envs, -1, 3).sum(1)[:, 0]).clamp_min(0)
        self.contact_all_step &= force > self.contact_threshold_N
        dj, dp, growth = self.events.advance(force, self.wall_present)
        self.event_impulse_step += dj
        self.event_peak_step += dp
        # Average over the control interval: reward dt then integrates this cost.
        self.event_growth_step += growth / self.env.decimation

    def task_contact_state(self):
        target = self.target_ee_world() - self.env.scene.env_origins
        in_region = (target[:, 1] >= -.35) & (target[:, 1] <= .65) & (target[:, 2] >= .5) & (target[:, 2] <= 1.5)
        required = self.wall_present & in_region & (target[:, 0] > self.wall_x + self.contact_target_margin_m)
        allowed = self.wall_present & in_region & (target[:, 0] >= self.wall_x - .02)
        return required, allowed

    def contact_requirement_state(self):
        required, _ = self.task_contact_state()
        return required, self.wall_contact_this_step

    @reward
    def gentle_tracking(self):
        error = self.target_ee_world() - self.asset.data.body_pos_w[:, self.hand_asset]
        required, _ = self.task_contact_state()
        # Mode follows the nominal task, never the policy's decision to touch.
        # Tangential tracking + the independent contact requirement enforce the
        # feasible wall task without pushing the hand toward an interior point.
        error[:, 0] *= (~required).float()
        return _calc_exp_sigma(error.norm(dim=-1, keepdim=True), self.reward_sigma['ee'])

    @reward
    def gentle_requirement(self):
        required, allowed = self.task_contact_state()
        self.contact_required_age = torch.where(required, self.contact_required_age+1, 0)
        ramp = ((self.contact_required_age-self.contact_grace_steps).clamp_min(0).float()
                / self.contact_miss_ramp_steps).clamp(max=1.)
        missing = required & ~self.wall_contact_this_step
        unexpected = ~allowed & self.wall_contact_this_step
        return (-self.contact_late_penalty_max*ramp*missing
                -self.unexpected_contact_score*unexpected).unsqueeze(-1)

    @reward
    def gentle_approach(self):
        hand = self.asset.data.body_pos_w[:, self.hand_asset]
        distance = self.wall_x + self.env.scene.env_origins[:, 0] - hand[:, 0]
        width = self.gentle.get('approach_distance', .1)
        gate = (1.-distance/width).clamp(0., 1.)
        gate = gate.square()*(3.-2.*gate)
        target = self.target_ee_world() - self.env.scene.env_origins
        relevant = (self.wall_present & (target[:, 0] >= self.wall_x-width)
                    & (target[:, 1] >= -.35) & (target[:, 1] <= .65)
                    & (target[:, 2] >= .5) & (target[:, 2] <= 1.5)
                    & ~self.wall_contact_this_step & (distance > 0))
        velocity = self.asset.data.body_lin_vel_w[:, self.hand_asset, 0].clamp_min(0)
        cost = ((velocity-.1).clamp_min(0)/.2).square().clamp(max=.5)
        return (-self.gentle_factor*gate*cost*relevant).unsqueeze(-1)

    @reward
    def gentle_success(self):
        required, _ = self.task_contact_state()
        tangent_error = (self.target_ee_world()-self.asset.data.body_pos_w[:, self.hand_asset])[:, 1:].norm(dim=-1)
        good = required & self.contact_all_step & (tangent_error <= .03)
        self.hold_steps = torch.where(good, self.hold_steps+1, 0)
        success = (self.hold_steps >= round(.3/self.env.step_dt)) & ~self.success_paid
        self.success_paid |= success
        # Actual event bonus .2, independent of the control timestep.
        return (success.float() * (.2/self.env.step_dt)).unsqueeze(-1)

    @reward
    def gentle_residual_smooth(self):
        actions = self.env.action_manager.high_action_buf
        scale = self.env.action_manager.ee_pos_scale[:, 0]
        change = (torch.tanh(actions[:, 0])-torch.tanh(actions[:, 1]))*scale
        return (-.01*(change/.02).square().sum(-1).clamp(max=5.)).unsqueeze(-1)

    @reward
    def gentle_impulse(self):
        return (-self.gentle_factor*self.event_impulse_step).unsqueeze(-1)

    @reward
    def gentle_growth(self):
        return (-self.gentle_factor*self.event_growth_step).unsqueeze(-1)

    @reward
    def gentle_peak(self):
        return (-self.gentle_factor*self.event_peak_step).unsqueeze(-1)

    def before_update(self):
        super().before_update()
        if hasattr(self, 'events') and self.cloud_counter % 16 == 0:
            occurred = self.events.count > 0
            j = self.events.impulse[occurred]
            self.env.extra.update({
                'gentle/curriculum': self.gentle_factor,
                'gentle/success_fraction': self.success_paid.float().mean().item(),
                'gentle/events_per_env': self.events.count.float().mean().item(),
                'gentle/repeat_contact_fraction': (self.events.count > 1).float().mean().item(),
                'gentle/impulse_exceed_fraction': (j > 2.).float().mean().item() if j.numel() else 0.,
                'gentle/impulse_p95_Ns': torch.quantile(j, .95).item() if j.numel() else 0.,
                'gentle/approach_penalty': -self.gentle_approach().mean().item(),
            })
