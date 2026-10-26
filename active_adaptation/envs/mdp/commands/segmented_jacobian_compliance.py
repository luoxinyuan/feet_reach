"""Frozen CEER residual training with offline optimized reference force prompts."""
import json
from pathlib import Path
import numpy as np
import torch
from .expanded_spring_compliance import ExpandedSpringComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.utils.body_guidance import slew_force
from active_adaptation.utils.point_guidance import reduce_point_forces


class SegmentedJacobianComplianceCommand(ExpandedSpringComplianceCommand):
    def __init__(self,env,*args,guidance_cache,**kwargs):
        super().__init__(env,*args,**kwargs)
        path=Path(guidance_cache)
        if not path.is_absolute():path=Path(__file__).resolve().parents[4]/path
        manifest=json.loads((path/'manifest.json').read_text())
        if [p['name'] for p in manifest['points']]!=self.point_names:
            raise ValueError('Force cache point layout does not match task config')
        import hashlib
        if hashlib.sha256(Path(self.asset.cfg.spawn.usd_path).read_bytes()).hexdigest()!=manifest['geometry_sha256']:
            raise ValueError('Force cache robot geometry differs from training USD')
        self.cache_slots=torch.as_tensor(np.load(path/'slots.npy').astype(np.int64),device=self.device)
        self.cache_forces=torch.as_tensor(np.load(path/'forces.npy'),device=self.device)
        self.cache_offsets=torch.as_tensor(np.load(path/'motion_offsets.npy'),device=self.device)
        self.point_desired=torch.zeros_like(self.point_force_buffer)
        self.cache_index=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)

    def _select(self,ids):
        # Selection has already been optimized using complete reference clips.
        pass

    def _solve(self):
        mid=self._motion.motion_id[:,0].long();step=self._motion.step[:,0].long()
        offset=self.cache_offsets[mid]
        if bool((offset<0).any()):raise RuntimeError('Motion missing from guidance cache')
        self.cache_index=offset+step
        slots=self.cache_slots[self.cache_index]
        self.active.copy_(slots>=0);self.slots.copy_(slots.clamp_min(0))
        self.desired.copy_(self.cache_forces[self.cache_index])
        if not self.guidance_enabled:self.desired.zero_()
        self.point_desired.zero_()
        self.point_desired.scatter_add_(1,self.slots[...,None].expand_as(self.desired),self.desired)

    def reset(self,ids):
        # Parent resets sampled motion/targets and calls our reference/solve hooks.
        super().reset(ids)
        self.point_force_buffer[ids]=0

    def before_update(self):
        MotionTrackingCommand.before_update(self)
        self.update_reward_target();self._reference();self._solve();self.log_step+=1
        if self.log_step%16==0:
            norm=self.point_force_buffer.norm(dim=-1)
            self.env.extra.update({
                'guidance/force_max_N':norm.max().item(),
                'guidance/force_budget_max_N':norm.sum(-1).max().item(),
                'guidance/contact_count':(norm>1e-5).sum(-1).float().mean().item(),
                'guidance/cache_force_error_N':(self.point_force_buffer-self.point_desired).norm(dim=-1).mean().item(),
                'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.all_joint_idx_asset]-self._motion.joint_pos[:,0,self.all_joint_idx_dataset]).abs().mean().item()})
            for i,name in enumerate(self.point_names):
                self.env.extra[f'guidance/point_active/{name}']=(norm[:,i]>1e-5).float().mean().item()

    def step(self,substep):
        self.point_force_buffer.copy_(slew_force(self.point_force_buffer,self.point_desired,self.env.physics_dt))
        self.force.copy_(self.point_force_buffer.gather(1,self.slots[...,None].expand_as(self.force)))
        self.force.mul_(self.active[...,None])
        ids=self.point_ids[None].expand(self.num_envs,-1)
        forces,moments=reduce_point_forces(self.point_force_buffer,self.point_positions_w(),ids,self.asset.data.body_pos_w)
        self.force_apply_buffer.copy_(forces);self.torque_apply_buffer.copy_(moments)
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.asset.has_external_wrench=False
