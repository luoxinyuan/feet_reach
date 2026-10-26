"""Controlled dense+AMP ablation; only PPO rewards gain a motion prior.

Discriminator is training-only, with fixed expert normalization and manually
averaged gradients across ranks (including the gradient-penalty gradients).
"""
from pathlib import Path
import torch
from torch import nn
from .sonic_residual_ppo import SonicResidualPPO
from active_adaptation.utils.compliance_amp import pair_features,style_reward

class MotionDiscriminator(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1))
    def normalize(self,x):return ((x-self.mean)/self.scale).clamp(-10,10)
    def forward(self,x):return self.net(self.normalize(x))

class ComplianceAMPPPO(SonicResidualPPO):
    def __init__(self,cfg,*args,env=None,**kwargs):
        super().__init__(cfg,*args,env=env,**kwargs)
        dim=2*env.observation_spec['amp_state_'].shape[-1]
        # Keep actor/critic initialization and public observation dimensions identical.
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.manual_seed(20261024)
            self.discriminator=MotionDiscriminator(dim).to(self.device)
        if torch.distributed.is_initialized():
            for p in self.discriminator.parameters():torch.distributed.broadcast(p.data,0)
        self.opt_disc=torch.optim.Adam(self.discriminator.parameters(),lr=cfg.amp_lr,weight_decay=1e-4)
        self.amp_dt=float(env.step_dt);self.expert=None
        self.replay=torch.empty(cfg.amp_replay_size,dim,device=self.device)
        self.replay_count=0;self.replay_cursor=0

    def _ensure_expert(self):
        if self.expert is not None:return
        path=Path(self.cfg.amp_expert)
        if not path.is_absolute():path=Path(__file__).resolve().parents[3]/path
        state=torch.load(path,map_location='cpu',weights_only=False)
        self.expert=state['pairs'].to(self.device)
        self.discriminator.mean.copy_(state['mean']);self.discriminator.scale.copy_(state['scale'])

    def _push_replay(self,pairs):
        size=min(len(pairs),self.cfg.amp_batch_size*4,len(self.replay))
        if size==0:return
        chosen=pairs[torch.randint(len(pairs),(size,),device=self.device)]
        ids=(torch.arange(size,device=self.device)+self.replay_cursor)%len(self.replay)
        self.replay[ids]=chosen;self.replay_cursor=(self.replay_cursor+size)%len(self.replay)
        self.replay_count=min(len(self.replay),self.replay_count+size)

    def train_op(self,td,vecnorm):
        self._ensure_expert()
        current=td['amp_state_'];following=td['next','amp_state_']
        pairs=pair_features(current,following).reshape(-1,current.shape[-1]*2).detach()
        # Ignore initialization and terminal transitions. Samples are one physical
        # step, never a concatenation of pre-reset and post-reset episodes.
        valid=(~td['is_init'].reshape(-1))&(~td['next','done'].reshape(-1))
        style=torch.zeros(len(pairs),1,device=self.device)
        with torch.no_grad():
            for lo in range(0,len(pairs),8192):style[lo:lo+8192]=style_reward(self.discriminator(pairs[lo:lo+8192]))
            style*=valid[:,None]
        weight=self.cfg.amp_weight*min(1.,self.num_updates/max(1,self.cfg.amp_warmup_updates))
        dense=td['next','reward'].clone()
        bonus=style.reshape(*td.batch_size,1)*weight*self.amp_dt
        mixed=dense.clone();mixed[...,:1]+=bonus
        td['next','reward']=mixed
        fake_current=pairs[valid]
        if len(fake_current)==0:raise RuntimeError('No valid AMP transitions in rollout')
        self._push_replay(fake_current)
        metrics=[]
        for _ in range(self.cfg.amp_disc_updates):
            n=self.cfg.amp_batch_size
            real=self.expert[torch.randint(len(self.expert),(n,),device=self.device)]
            fresh=fake_current[torch.randint(len(fake_current),(n//2,),device=self.device)]
            replay=self.replay[torch.randint(self.replay_count,(n-n//2,),device=self.device)]
            fake=torch.cat((fresh,replay))
            normalized=self.discriminator.normalize(real).detach().requires_grad_(True)
            real_score=self.discriminator.net(normalized);fake_score=self.discriminator(fake)
            classification=.5*((real_score-1).square().mean()+(fake_score+1).square().mean())
            grad=torch.autograd.grad(real_score.sum(),normalized,create_graph=True)[0]
            gp=grad.square().sum(-1).mean()
            loss=classification+self.cfg.amp_gradient_penalty*gp
            self.opt_disc.zero_grad(set_to_none=True);loss.backward()
            if torch.distributed.is_initialized():
                for p in self.discriminator.parameters():
                    torch.distributed.all_reduce(p.grad);p.grad.div_(torch.distributed.get_world_size())
            nn.utils.clip_grad_norm_(self.discriminator.parameters(),10.)
            self.opt_disc.step()
            metrics.append(torch.stack((loss.detach(),classification.detach(),gp.detach(),
                (real_score>0).float().mean(),(fake_score<0).float().mean())))
        stats=torch.stack(metrics).mean(0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats);stats/=torch.distributed.get_world_size()
        info=super().train_op(td,vecnorm)
        info.update(dict(zip(['amp/disc_loss','amp/classification_loss','amp/gradient_penalty','amp/expert_accuracy','amp/policy_accuracy'],stats.tolist())))
        info.update({'amp/style_reward':style.mean().item(),'amp/weight':weight,
            'amp/bonus_per_step':bonus.mean().item(),'amp/dense_reward_per_step':dense.sum(-1).mean().item(),
            'amp/valid_fraction':valid.float().mean().item(),'amp/replay_count':self.replay_count})
        return info

    def state_dict(self):
        state=super().state_dict();state['amp_discriminator']=self.discriminator.state_dict();state['optimizer_amp']=self.opt_disc.state_dict()
        return state

    def load_state_dict(self,state_dict,strict=True):
        super().load_state_dict(state_dict,strict=strict)
        if 'amp_discriminator' in state_dict:self.discriminator.load_state_dict(state_dict['amp_discriminator'])
        if 'optimizer_amp' in state_dict:self.opt_disc.load_state_dict(state_dict['optimizer_amp'])
        # Replay is deliberately rebuilt after resume; discriminator and optimizer persist.
