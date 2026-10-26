"""Residual PPO with a public-only actor and synchronized multi-GPU LR."""
import torch
from tensordict.nn import TensorDictSequential
from .root_ppo import RootPPOPolicy

class SonicResidualPPO(RootPPOPolicy):
    def __init__(self, cfg, *args, **kwargs):
        if list(cfg.in_keys) != ['policy']:
            raise ValueError('Residual student must only consume policy observations.')
        super().__init__(cfg, *args, **kwargs)

    def _do_lr_schedule(self, kl):
        # RootPPO's rank-local KL scheduler would eventually desynchronize Adam.
        if torch.distributed.is_initialized():
            average = torch.tensor(kl, device=self.device)
            torch.distributed.all_reduce(average)
            kl = (average / torch.distributed.get_world_size()).item()
        # Adapt from the first update; a frozen controller does not justify
        # RootPPO's initial 10%-of-training delay in KL protection.
        if kl > 2 * self.cfg.desired_kl:
            self.current_lr = max(1e-6, self.current_lr / 1.5)
        elif 0 < kl < .5 * self.cfg.desired_kl:
            self.current_lr = min(3e-4, self.current_lr * 1.5)
        for optimizer in (self.opt_actor, self.opt_critic):
            for group in optimizer.param_groups:
                group['lr'] = self.current_lr

    def get_rollout_policy(self, mode='train'):
        # scripts/train.py evaluates only rank 0 at the end. Its inference
        # forward must not invoke DDP collectives after other workers exit.
        actor = self.actor
        if mode in {'eval', 'deploy'} and isinstance(actor, torch.nn.parallel.DistributedDataParallel):
            actor = actor.module
        return TensorDictSequential(actor)

    def state_dict(self):
        state = super().state_dict()
        state['optimizer_actor'] = self.opt_actor.state_dict()
        state['optimizer_critic'] = self.opt_critic.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        for key, optimizer in (('optimizer_actor', self.opt_actor), ('optimizer_critic', self.opt_critic)):
            if key in state_dict:
                optimizer.load_state_dict(state_dict[key])
            for group in optimizer.param_groups:
                group['lr'] = self.current_lr
