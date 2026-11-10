"""Same residual PPO with a direct reference-increment oracle."""
from .root_ppo import RootPPOPolicy
from .sonic_residual_ppo import SonicResidualPPO

class GTDeltaQTeacherPPO(SonicResidualPPO):
    def __init__(self, cfg, *args, **kwargs):
        if list(cfg.in_keys) != ['policy', 'delta_q_gt_']:
            raise ValueError('Actor must use proprioception/self-reference and GT delta q only')
        RootPPOPolicy.__init__(self, cfg, *args, **kwargs)
