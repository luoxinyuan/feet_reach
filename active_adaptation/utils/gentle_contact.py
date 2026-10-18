"""Physics-rate contact events; no simulator dependencies."""
import torch


class ContactEvents:
    """Charge increments of event costs so early termination cannot erase them.

    A window starts on physical contact, including early permitted contact, and
    rearms only after sustained separation. The sum of cost increments equals
    the final event cost (before discounting/curriculum and reward dt scaling).
    """
    def __init__(self, n, device, dt, window=.1, release=.1):
        self.dt = dt
        self.window_steps = round(window / dt)
        self.release_steps = round(release / dt)
        if min(self.window_steps, self.release_steps) < 1:
            raise ValueError('Contact windows must be at least one physics step')
        if abs(self.window_steps * dt - window) > 1e-6:
            raise ValueError('Contact window must align with physics dt')
        self.armed = torch.ones(n, dtype=torch.bool, device=device)
        self.age = torch.full((n,), self.window_steps, dtype=torch.long, device=device)
        self.off_steps = torch.zeros_like(self.age)
        self.force_previous = torch.zeros(n, device=device)
        self.impulse = torch.zeros_like(self.force_previous)
        self.peak = torch.zeros_like(self.force_previous)
        self.impulse_paid = torch.zeros_like(self.force_previous)
        self.peak_paid = torch.zeros_like(self.force_previous)
        self.count = torch.zeros_like(self.age)

    def reset(self, ids):
        self.armed[ids] = True
        self.age[ids] = self.window_steps
        for x in (self.off_steps, self.force_previous, self.impulse, self.peak,
                  self.impulse_paid, self.peak_paid, self.count):
            x[ids] = 0

    def advance(self, force, wall_present, impulse_free=2., impulse_ref=3.1,
                impulse_scale=6., impulse_cap=10., peak_soft=40., peak_scale=60.,
                peak_weight=3., peak_cap=5., rate_soft=500., rate_scale=5000., rate_cap=.5):
        self.off_steps = torch.where(force < .5, self.off_steps + 1, 0)
        self.armed |= (self.off_steps >= self.release_steps) & (self.age >= self.window_steps)
        start = self.armed & wall_present & (force > 1.)
        for x in (self.age, self.impulse, self.peak, self.impulse_paid, self.peak_paid):
            x[start] = 0
        self.armed[start] = False
        self.count += start.long()
        active = (self.age < self.window_steps) & wall_present
        # Contact sensor samples are step-average force: include the onset step.
        self.impulse += force * self.dt * active
        self.peak = torch.where(active, torch.maximum(self.peak, force), self.peak)
        total_j = (impulse_scale * ((self.impulse-impulse_free).clamp_min(0)
                                    / (impulse_ref-impulse_free)).square()).clamp(max=impulse_cap)
        total_p = (peak_weight * ((self.peak-peak_soft).clamp_min(0)
                                  / peak_scale).square()).clamp(max=peak_cap)
        delta_j = (total_j-self.impulse_paid).clamp_min(0) * active
        delta_p = (total_p-self.peak_paid).clamp_min(0) * active
        self.impulse_paid = torch.where(active, total_j, self.impulse_paid)
        self.peak_paid = torch.where(active, total_p, self.peak_paid)
        rate = ((force-self.force_previous)/self.dt).clamp_min(0)
        growth = (((rate-rate_soft).clamp_min(0)/rate_scale).square()).clamp(max=rate_cap) * active
        self.force_previous.copy_(force)
        self.age += active.long()
        return delta_j, delta_p, growth
