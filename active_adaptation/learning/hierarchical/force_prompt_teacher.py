"""Same residual PPO teacher, without the unused student force estimator."""
from .root_ppo import RootPPOPolicy
from .sonic_residual_ppo import SonicResidualPPO

class ForcePromptTeacherPPO(SonicResidualPPO):
    def __init__(self,cfg,*args,**kwargs):
        if list(cfg.in_keys)!=['policy','force_gt_']:
            raise ValueError('Teacher actor may only use proprioception/self-reference and current force prompt')
        RootPPOPolicy.__init__(self,cfg,*args,**kwargs)
