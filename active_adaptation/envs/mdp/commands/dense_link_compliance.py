"""All-link cached reference prompts and causal force-estimator observations."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .sonic_compliance import SonicComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.body_guidance import slew_force

class DenseLinkComplianceCommand(SonicComplianceCommand):
    def __init__(self,env,*args,guidance_cache,estimator_history=10,force_slew=150.,**kwargs):
        super().__init__(env,*args,**kwargs)
        root=Path(__file__).resolve().parents[4]
        path=Path(guidance_cache);path=path if path.is_absolute() else root/path
        manifest=json.loads((path/'manifest.json').read_text())
        if abs(manifest['dt']-env.step_dt)>1e-8:raise ValueError('Force cache requires 50 Hz')
        paths=list(env.cfg.command.dataset.mem_paths)
        if len(paths)!=1:raise ValueError('Dense cache requires one explicit dataset split')
        source=Path(paths[0]);source=source if source.is_absolute() else root/'dataset'/source
        if hashlib.sha256((source/'meta_motion.json').read_bytes()).hexdigest()!=manifest['source_meta_sha256']:
            raise ValueError('Force cache and dataset view mismatch')
        self.point_names=manifest['body_names'];self.force_slew=float(force_slew)
        self.point_ids=torch.tensor([self.asset.body_names.index(n) for n in self.point_names],device=self.device)
        self.cache_offsets=torch.as_tensor(np.load(path/'motion_offsets.npy'),device=self.device)
        self.cache_forces=torch.as_tensor(np.load(path/'forces.npy'),device=self.device)
        self.point_force_buffer=torch.zeros(self.num_envs,len(self.point_names),3,device=self.device)
        self.point_desired=torch.zeros_like(self.point_force_buffer)
        self.estimator_history_steps=int(estimator_history)
        self.force_history=torch.zeros(self.num_envs,self.estimator_history_steps,93,device=self.device)
        self.history_fresh=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.history_tick=-1
        self.cache_manifest=manifest

    def _reference(self):pass
    def _select(self,ids):pass

    def _solve(self):
        mid=self._motion.motion_id[:,0].long();step=self._motion.step[:,0].long()
        offsets=self.cache_offsets[mid]
        if bool((offsets<0).any()):raise ValueError('Unknown original motion ID')
        self.point_desired.copy_(self.cache_forces[offsets+step])
        if not self.guidance_enabled:self.point_desired.zero_()

    def reset(self,ids):
        super().reset(ids)
        self.point_force_buffer[ids]=0
        self.force_history[ids]=0;self.history_fresh[ids]=True

    def before_update(self):
        MotionTrackingCommand.before_update(self)
        self.update_reward_target();self._solve();self.log_step+=1
        if self.log_step%16==0:
            norm=self.point_force_buffer.norm(dim=-1)
            self.env.extra.update({'guidance/k_N_per_m':self.cache_manifest['k_N_per_m'],
                'guidance/max_point_N':norm.max().item(),
                'guidance/mean_sum_N':norm.sum(-1).mean().item(),
                'guidance/max_sum_N':norm.sum(-1).max().item(),
                'guidance/max_net_N':self.point_force_buffer.sum(1).norm(dim=-1).max().item(),
                'guidance/actual_vs_nominal_N':(self.point_force_buffer-self.point_desired).norm(dim=-1).mean().item()})

    def step(self,substep):
        self.point_force_buffer.copy_(slew_force(self.point_force_buffer,self.point_desired,self.env.physics_dt,self.force_slew))
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        self.force_apply_buffer[:,self.point_ids]=self.point_force_buffer
        # These are link origins, not joints/actuator torques.
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.asset.has_external_wrench=False

    @observation
    def dense_force_gt(self):
        q=self.asset.data.root_quat_w[:,None].expand(-1,len(self.point_names),-1)
        # Actual physically applied force, including slew/reset transients.
        return quat_apply_inverse(q,self.point_force_buffer).flatten(1)/30.

    def dense_force_gt_sym(self):
        n=3*len(self.point_names);return SymmetryTransform(torch.arange(n),torch.ones(n))

    @observation
    def dense_estimator_history(self):
        manager=self.env.action_manager;data=self.asset.data
        current=torch.cat((data.root_ang_vel_b/5.,data.projected_gravity_b,
            data.joint_pos[:,manager.joint_ids],data.joint_vel[:,manager.joint_ids]/10.,
            manager.low_action.clamp(-10,10)/5.),-1)
        if self.history_tick!=self.log_step:
            self.force_history[:,1:]=self.force_history[:,:-1].clone()
            self.force_history[:,0]=current;self.history_tick=self.log_step
        self.force_history[self.history_fresh]=current[self.history_fresh,None]
        self.history_fresh.zero_()
        return self.force_history.flatten(1)

    def dense_estimator_history_sym(self):
        n=self.estimator_history_steps*93;return SymmetryTransform(torch.arange(n),torch.ones(n))
