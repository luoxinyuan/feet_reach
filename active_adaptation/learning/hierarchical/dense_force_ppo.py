"""GT-force residual teacher -> causal adaptation -> estimated-force residual PPO.

The estimator follows CEER's direct privileged adaptation MLP (512/256 + MSE),
expanded to a vector per link. It is not the frozen low-level state estimator.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod,TensorDictSequential as Seq
from .root_ppo import RootPPOPolicy
from .sonic_residual_ppo import SonicResidualPPO
from active_adaptation.learning.ppo.common import make_mlp,make_batch
from active_adaptation.learning.modules.distributions import IndependentNormal


def plain(module):
    return module.module if isinstance(module,torch.nn.parallel.DistributedDataParallel) else module

class ForceEstimator(nn.Module):
    def __init__(self,dim,norm):
        super().__init__();self.net=nn.Sequential(make_mlp([512,256],norm=norm),nn.LazyLinear(dim))
    def forward(self,x):
        result=self.net(x);forces=result.reshape(*result.shape[:-1],-1,3)
        return (forces/forces.norm(dim=-1,keepdim=True).clamp_min(1.)).flatten(-2)

class DenseForcePPO(RootPPOPolicy):
    def __init__(self,cfg,observation_spec,*args,**kwargs):
        self.phase=cfg.phase
        if self.phase not in ('teacher','distill','finetune'):raise ValueError(self.phase)
        if list(cfg.in_keys)!=['policy','force_input_']:raise ValueError('Unaudited actor inputs')
        observation_spec=observation_spec.clone()
        observation_spec['force_input_']=observation_spec['force_gt_'].clone()
        super().__init__(cfg,observation_spec,*args,**kwargs)
        self.teacher=deepcopy(plain(self.actor)).requires_grad_(False).eval()
        estimator=Mod(ForceEstimator(observation_spec['force_gt_'].shape[-1],cfg.layer_norm),['estimator_history_'],['force_input_']).to(self.device)
        estimator(observation_spec.zero().to(self.device))
        if torch.distributed.is_initialized():
            estimator=torch.nn.parallel.DistributedDataParallel(estimator,device_ids=[self.device.index],broadcast_buffers=False)
        self.estimator=estimator
        self.opt_estimator=torch.optim.Adam(self.estimator.parameters(),lr=cfg.estimator_lr)
        self.stage_initialized=self.phase=='teacher'

    def _do_lr_schedule(self,kl):return SonicResidualPPO._do_lr_schedule(self,kl)

    def get_rollout_policy(self,mode='train'):
        actor=plain(self.actor) if mode in ('eval','deploy') else self.actor
        if self.phase=='teacher':
            return Seq(Mod(lambda f:f.clone(),['force_gt_'],['force_input_']),actor)
        estimator=plain(self.estimator) if mode in ('eval','deploy') else self.estimator
        # No force_gt_, future target, reference motion or privileged fields.
        return Seq(estimator,actor)

    @torch.no_grad()
    def teacher_targets(self,td):
        # Bound transient activations when supervising large N x T rollouts.
        # Teacher is feed-forward with LayerNorm, so rows are independent.
        flat=td.reshape(-1)
        result=torch.empty(len(flat),self.action_dim,device=self.device,dtype=td['policy'].dtype)
        for start in range(0,len(flat),65536):
            batch=flat[start:start+65536]
            teacher_td=TensorDict({'policy':batch['policy'],'force_input_':batch['force_gt_']},batch.batch_size)
            self.teacher(teacher_td)
            result[start:start+len(batch)]=teacher_td['loc']
        return result.reshape(*td.batch_size,self.action_dim)

    def _estimator_update(self,td):
        values=[]
        for _ in range(self.cfg.estimator_epochs):
            for mb in make_batch(td,self.num_minibatches):
                out=self.estimator(mb.select('estimator_history_').clone())['force_input_']
                valid=(~mb['is_init']).float();error=(out-mb['force_gt_']).square().mean(-1,keepdim=True)
                loss=(error*valid).sum()/valid.sum().clamp_min(1)
                self.opt_estimator.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(),1.);self.opt_estimator.step()
                values.append(loss.detach())
        return torch.stack(values).mean()

    def train_op(self,td,vecnorm):
        if not self.stage_initialized:raise RuntimeError('Student stages require a preceding-stage checkpoint')
        if self.phase=='teacher':return super().train_op(td,vecnorm)
        td['teacher_loc_']=self.teacher_targets(td)
        if self.phase=='distill':
            losses=[]
            for _ in range(self.cfg.distill_epochs):
                for mb in make_batch(td,self.num_minibatches):
                    # DAgger: query teacher on student-visited states. Cached
                    # predicted forces are those used during this rollout.
                    out=self.actor(mb.select('policy','force_input_').clone())
                    valid=(~mb['is_init']).float()
                    loss=((out['loc'].tanh()-mb['teacher_loc_'].tanh()).square().mean(-1,keepdim=True)*valid).sum()/valid.sum().clamp_min(1)
                    loss=loss+0.*out['scale'].sum() # keep DDP log-std graph explicit
                    self.opt_actor.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.opt_actor.step()
                    losses.append(loss.detach())
            info={'adapt/imitation_loss':torch.stack(losses).mean().item()};self.num_updates+=1
        else:
            info=super().train_op(td,vecnorm)
        # Update estimator AFTER PPO: changing its predictions during PPO would
        # invalidate the stored behavior distribution/importance ratios.
        loss=self._estimator_update(td)
        info.update({'adapt/force_mse_scaled':loss.item(),'adapt/force_component_rmse_N':30*loss.sqrt().item()})
        return info

    def _update(self,mb):
        if self.phase=='teacher':return super()._update(mb)
        loc_old=mb['loc'].clone();scale_old=mb['scale'].clone();action_old=mb['action'].clone();logp_old=mb['sample_log_prob'].clone()
        valid=~mb['is_init'];mb=mb.exclude('next','sample_log_prob','action')
        self.actor(mb);values=self.critic(mb)['state_value']
        dist=IndependentNormal(mb['loc'],mb['scale']);logp=dist.log_prob(action_old);entropy=dist.entropy().mean()
        ratio=torch.exp(logp-logp_old).unsqueeze(-1)
        policy_loss=-torch.mean(torch.min(mb['adv']*ratio,mb['adv']*ratio.clamp(1-self.clip_param,1+self.clip_param))*valid)
        value_loss=(F.mse_loss(values,mb['ret'],reduction='none')*valid).mean()
        imitation=((mb['loc'].tanh()-mb['teacher_loc_'].tanh()).square().mean(-1,keepdim=True)*valid).mean()
        loss=policy_loss-self.entropy_coef*entropy+value_loss+self.cfg.imitation_weight*imitation
        self.opt_actor.zero_grad();self.opt_critic.zero_grad();loss.backward()
        actor_grad=nn.utils.clip_grad_norm_(self.actor.parameters(),1.);critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),1.)
        self.opt_actor.step();self.opt_critic.step()
        with torch.no_grad():
            kl=(torch.log(mb['scale']/scale_old)+(scale_old.square()+(loc_old-mb['loc']).square())/(2*mb['scale'].square())-.5).sum(-1).mean()
        return {'actor/policy_loss':policy_loss.detach(),'actor/entropy':entropy.detach(),'actor/actor_grad_norm':actor_grad.detach(),
            'actor/clamp_ratio':((ratio-1).abs()>self.clip_param).float().mean().detach(),'actor/kl':kl.detach(),
            'critic/value_loss':value_loss.detach(),'critic/critic_grad_norm':critic_grad.detach(),'adapt/imitation_loss':imitation.detach()}

    def state_dict(self):
        state=super().state_dict()
        state.update(phase=self.phase,estimator=plain(self.estimator).state_dict(),teacher=self.teacher.state_dict(),
            optimizer_actor=self.opt_actor.state_dict(),optimizer_critic=self.opt_critic.state_dict(),optimizer_estimator=self.opt_estimator.state_dict())
        return state

    def load_state_dict(self,state_dict,strict=True):
        source=state_dict.get('phase')
        if source is None:raise ValueError('Expected a dense-force staged checkpoint')
        if source!=self.phase and (source,self.phase) not in [('teacher','distill'),('distill','finetune')]:
            raise ValueError(f'Invalid stage transition {source} -> {self.phase}')
        super().load_state_dict(state_dict,strict=strict)
        plain(self.estimator).load_state_dict(state_dict['estimator'])
        self.teacher.load_state_dict(state_dict['actor'] if source=='teacher' else state_dict['teacher'])
        if source==self.phase:
            for name in ('actor','critic','estimator'):
                getattr(self,'opt_'+name).load_state_dict(state_dict['optimizer_'+name])
        else:
            self.num_updates=0;self.progress=0.;self.current_lr=self.cfg.lr;self.entropy_coef=self.cfg.entropy_coef_start
        self.stage_initialized=True
