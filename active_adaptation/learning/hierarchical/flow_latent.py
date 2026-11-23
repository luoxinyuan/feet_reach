"""Causal single-latent flow distilled from on-policy PPO experience.

The PPO collector remains the behavior policy. The flow is a separate deployable
policy, NOT an actor trained with Gaussian PPO likelihoods. Only the current executed action
is used as supervision; no future action labels are read. SuperMimic remains frozen in the env.
"""
import torch
from torch import nn
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.utils import exploration_type, ExplorationType
from .force_prompt_teacher import ForcePromptTeacherPPO


def plain(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


class LatentFlow(nn.Module):
    def __init__(self, condition_dim, action_dim=32, horizon=1, width=512):
        super().__init__()
        self.action_dim, self.horizon = action_dim, horizon
        self.condition = nn.Sequential(nn.Linear(condition_dim, width), nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 256), nn.SiLU())
        self.velocity = nn.Sequential(nn.Linear(256 + action_dim*horizon + 1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, action_dim*horizon))

    def forward(self, condition, x, time):
        return self.velocity(torch.cat([self.condition(condition), x, time], -1))

    @torch.no_grad()
    def sample(self, condition, steps=8, noise=None):
        x = torch.randn(*condition.shape[:-1], self.action_dim*self.horizon, device=condition.device) if noise is None else noise.clone()
        for i in range(steps):
            time = torch.full_like(x[..., :1], i/steps)
            x = x + self(condition, x, time)/steps
        return x.reshape(*condition.shape[:-1], self.horizon, self.action_dim).clamp(-0.999, 0.999)


class FlowDeployment(TensorDictModuleBase):
    in_keys = ['policy', 'force_gt_']
    out_keys = ['action']

    def __init__(self, flow, steps):
        super().__init__()
        self.flow, self.steps = flow, steps
        self.noise = None

    def forward(self, td):
        condition = torch.cat([td[k] for k in self.in_keys], -1)
        shape = (*condition.shape[:-1], self.flow.action_dim*self.flow.horizon)
        # A persistent noise seed avoids independently switching modes at 50 Hz.
        if exploration_type() == ExplorationType.MODE:
            generator = torch.Generator(device=condition.device).manual_seed(0)
            noise = torch.randn(shape, device=condition.device, generator=generator)
        else:
            if self.noise is None or self.noise.shape != shape:
                self.noise = torch.randn(shape, device=condition.device)
            if 'is_init' in td:
                reset = td['is_init'].squeeze(-1)
                self.noise[reset] = torch.randn_like(self.noise[reset])
            noise = self.noise
        chunk = self.flow.sample(condition, self.steps, noise)
        # Environment consumes raw actions through tanh, then latent_scale=2.
        td['action'] = torch.atanh(chunk[..., 0, :])
        return td


class FlowLatentPPO(ForcePromptTeacherPPO):
    def __init__(self, cfg, observation_spec, *args, **kwargs):
        super().__init__(cfg, observation_spec, *args, **kwargs)
        if self.action_dim != 32:
            raise ValueError('This configuration requires the 32-D latent interface')
        if cfg.flow_horizon != 1:
            raise ValueError('Single-latent training requires flow_horizon=1')
        dim = sum(observation_spec[k].shape[-1] for k in self.cfg.in_keys)
        self.flow = LatentFlow(dim, self.action_dim, cfg.flow_horizon, cfg.flow_width).to(self.device)
        if torch.distributed.is_initialized():
            self.flow = torch.nn.parallel.DistributedDataParallel(self.flow, device_ids=[self.device.index], broadcast_buffers=False)
        self.opt_flow = torch.optim.Adam(self.flow.parameters(), lr=cfg.flow_lr)

    def get_rollout_policy(self, mode='train'):
        if mode in ('eval', 'deploy'):
            return FlowDeployment(plain(self.flow), self.cfg.flow_steps)
        return super().get_rollout_policy(mode)

    def train_op(self, td, vecnorm):
        info = super().train_op(td, vecnorm)
        if self.num_updates < self.cfg.flow_warmup:
            return {**info, 'flow/warming_up': 1.}
        # Supervise only the action at the conditioning state; no future labels.
        valid = ~td['is_init'].squeeze(-1) & ~td['next', 'done'].squeeze(-1)
        indices = valid.nonzero()
        count = torch.tensor(len(indices), device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MIN)
        if count.item() == 0:
            return {**info, 'flow/valid_samples': 0.}
        losses=[]
        for _ in range(self.cfg.flow_updates):
            chosen = indices[torch.randint(len(indices), (self.cfg.flow_batch_size,), device=self.device)]
            rows, start = chosen.unbind(-1)
            # Executed PPO actions, not motion GT or privileged teacher inputs.
            target = td['action'][rows, start].detach().tanh()
            condition = torch.cat([td[k][rows, start] for k in self.cfg.in_keys], -1).detach()
            noise = torch.randn_like(target)
            tau = torch.rand(len(target), 1, device=self.device)
            pred = self.flow(condition, (1-tau)*noise+tau*target, tau)
            # Favor locally better-than-baseline rollout actions; cap weights.
            weights = td['adv'][rows, start].detach().clamp(-2, 2).exp()
            weights = weights/weights.mean().clamp_min(1e-6)
            loss = ((pred-(target-noise)).square().mean(-1, keepdim=True)*weights).mean()
            self.opt_flow.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.flow.parameters(), 1.)
            self.opt_flow.step()
            losses.append(loss.detach())
        return {**info, 'flow/loss': torch.stack(losses).mean().item(), 'flow/valid_samples': float(len(indices)), 'flow/warming_up': 0.}

    def state_dict(self):
        state = super().state_dict()
        state.update(flow=plain(self.flow).state_dict(), optimizer_flow=self.opt_flow.state_dict(), flow_format='single_latent_v1')
        return state

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get('flow_format') != 'single_latent_v1':
            raise ValueError('Expected single-latent checkpoint; do not load ten-step flow')
        super().load_state_dict(state_dict, strict)
        plain(self.flow).load_state_dict(state_dict['flow'], strict=strict)
        self.opt_flow.load_state_dict(state_dict['optimizer_flow'])
