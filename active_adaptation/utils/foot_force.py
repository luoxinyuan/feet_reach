"""Per-environment linear rest/up/hold/down force schedule (control steps)."""
import torch


class FootForceRamp:
    def __init__(self, count, device, force_range=(0.,20.), ramp_up_range=(25,100),
                 hold_range=(20,200), ramp_down_range=(25,100), rest_range=(20,200),
                 zero_prob=.1):
        self.device=device
        self.force_range=tuple(force_range)
        self.ranges=[tuple(rest_range),tuple(ramp_up_range),tuple(hold_range),tuple(ramp_down_range)]
        self.zero_prob=float(zero_prob)
        if not 0 <= self.force_range[0] <= self.force_range[1] <= 20:
            raise ValueError('Foot force range must be within 0..20 N')
        if not 0 <= self.zero_prob <= 1 or any(not 1 <= a <= b for a,b in self.ranges):
            raise ValueError('Invalid probability or phase duration')
        self.phase=torch.zeros(count,dtype=torch.long,device=device)
        self.elapsed=torch.zeros_like(self.phase)
        self.duration=torch.ones_like(self.phase)
        self.peak=torch.zeros(count,3,device=device)
        self.force=torch.zeros_like(self.peak)
        self.reset(torch.arange(count,device=device))

    def _duration(self, ids, phase):
        lo,hi=self.ranges[phase]
        self.duration[ids]=torch.randint(lo,hi+1,(len(ids),),device=self.device)

    def reset(self, ids):
        self.phase[ids]=0;self.elapsed[ids]=0
        self.peak[ids]=0;self.force[ids]=0
        self._duration(ids,0)

    def advance(self):
        # Direction and amplitude stay fixed throughout each up/hold/down cycle.
        finished=self.elapsed>=self.duration
        old=self.phase.clone()
        for phase in range(4):
            ids=(finished & (old==phase)).nonzero().flatten()
            if not len(ids): continue
            next_phase=(phase+1)%4
            self.phase[ids]=next_phase;self.elapsed[ids]=0
            self._duration(ids,next_phase)
            if next_phase==1:
                direction=torch.randn(len(ids),3,device=self.device)
                direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                lo,hi=self.force_range
                magnitude=lo+(hi-lo)*torch.rand(len(ids),1,device=self.device)
                magnitude*= (torch.rand(len(ids),1,device=self.device)>=self.zero_prob)
                self.peak[ids]=direction*magnitude
            elif next_phase==0:
                self.peak[ids]=0
        self.elapsed+=1
        alpha=(self.elapsed/self.duration).clamp(0,1)
        scale=torch.where(self.phase==1,alpha,torch.where(self.phase==2,1.,torch.where(self.phase==3,1.-alpha,0.)))
        self.force.copy_(self.peak*scale[:,None])
        return self.force
