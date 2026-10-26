"""Training-only actual motion observation; public actor inputs are unchanged."""
import torch
from .segmented_jacobian_compliance import SegmentedJacobianComplianceCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.compliance_amp import motion_features,AMP_BODIES
from active_adaptation.utils.symmetry import SymmetryTransform

class AMPComplianceCommand(SegmentedJacobianComplianceCommand):
    def __init__(self,env,*args,**kwargs):
        super().__init__(env,*args,**kwargs)
        self.amp_joint_ids=[self.asset.joint_names.index(n) for n in self.dataset.joint_names]
        self.amp_body_ids=[self.asset.body_names.index(n) for n in AMP_BODIES]

    @observation
    def amp_actual_motion(self):
        data=self.asset.data
        return motion_features(data.joint_pos[:,self.amp_joint_ids],data.joint_vel[:,self.amp_joint_ids],
            data.root_pos_w-self.env.scene.env_origins,data.root_quat_w,
            data.root_lin_vel_w,data.root_ang_vel_w,
            data.body_pos_w[:,self.amp_body_ids]-self.env.scene.env_origins[:,None])

    def amp_actual_motion_sym(self):
        return SymmetryTransform(torch.arange(83),torch.ones(83))
