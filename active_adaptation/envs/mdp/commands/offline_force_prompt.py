"""Offline force tokens, optionally also applied at the robot's contact points."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .sonic_compliance import SonicComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply_inverse, quat_apply
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.sonic import JOINT_NAMES

class OfflineForcePromptCommand(SonicComplianceCommand):
    def __init__(self,env,*args,guidance_cache,physical_guidance=False,**kwargs):
        super().__init__(env,*args,**kwargs)
        path=Path(guidance_cache)
        if not path.is_absolute():path=Path(__file__).resolve().parents[4]/path
        manifest=json.loads((path/'manifest.json').read_text())
        if not manifest.get('complete'):raise ValueError('Force cache must be complete')
        if abs(manifest['dt']-env.step_dt)>1e-8:raise ValueError('Force prompt requires 50 Hz')
        if list(env.cfg.command.dataset.mem_paths)!=manifest['dataset_paths']:raise ValueError('Force cache/dataset mismatch')
        for source,digest in zip(manifest['dataset_paths'],manifest['dataset_meta_sha256']):
            if hashlib.sha256((Path(source)/'meta_motion.json').read_bytes()).hexdigest()!=digest:raise ValueError('Dataset metadata changed')
        if hashlib.sha256(Path(self.asset.cfg.spawn.usd_path).read_bytes()).hexdigest()!=manifest['geometry']['sha256']:raise ValueError('Force cache robot geometry mismatch')
        self.reference_target_ids=[self.dataset.joint_names.index(n) for n in JOINT_NAMES]
        self.cache_manifest=manifest
        self.point_names=[p['name'] for p in manifest['points']]
        self.cache_offsets=torch.as_tensor(np.load(path/'motion_offsets.npy'),device=self.device)
        self.cache_forces=torch.as_tensor(np.load(path/'forces.npy'),device=self.device)
        self.point_force_buffer=torch.zeros(self.num_envs,len(self.point_names),3,device=self.device)
        self.physical_guidance=bool(physical_guidance)
        self.prompt_body_ids=torch.tensor([self.asset.body_names.index(p['body']) for p in manifest['points']],device=self.device)
        self.prompt_offsets=torch.tensor([p['offset'] for p in manifest['points']],device=self.device,dtype=torch.float32)
        self.force_apply_world=self.physical_guidance

    def _reference(self):pass
    def _select(self,ids):pass

    def _solve(self):
        mid=self._motion.motion_id[:,0].long();step=self._motion.step[:,0].long()
        self.point_force_buffer.copy_(self.cache_forces[self.cache_offsets[mid]+step])
        # At the clip end no next transition exists. Also handles dataset truncation.
        self.point_force_buffer[self.t>=self.lengths-1]=0
        if not self.guidance_enabled:self.point_force_buffer.zero_()

    def before_update(self):
        MotionTrackingCommand.before_update(self)
        self.update_reward_target();self._solve();self.log_step+=1
        if self.log_step%16==0:
            norm=self.point_force_buffer.norm(dim=-1)
            self.env.extra.update({'prompt/mean_sum_N':norm.sum(-1).mean().item(),
                'prompt/max_sum_N':norm.sum(-1).max().item(),'prompt/max_point_N':norm.max().item(),
                'prompt/physical_force_max_N':self.force_apply_buffer.norm(dim=-1).max().item(),
                'tracking/reference_joint_mae_rad':(self.env.action_manager.reference.q-self._motion.joint_pos[:,0,self.reference_target_ids]).abs().mean().item(),
                'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.all_joint_idx_asset]-self._motion.joint_pos[:,0,self.all_joint_idx_dataset]).abs().mean().item()})

    def step(self,substep):
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        self.asset.has_external_wrench=False
        if self.physical_guidance:
            # Several points may share one rigid body. Apply the equivalent
            # resultant at its link origin plus the offset-induced moment.
            # Recompute rotated offsets every physics step; the cached force
            # remains world-frame and is held over the control interval.
            ids=self.prompt_body_ids[None,:,None].expand(self.num_envs,-1,3)
            offsets=quat_apply(self.asset.data.body_quat_w[:,self.prompt_body_ids],
                               self.prompt_offsets[None].expand(self.num_envs,-1,-1))
            self.force_apply_buffer.scatter_add_(1,ids,self.point_force_buffer)
            self.torque_apply_buffer.scatter_add_(1,ids,torch.cross(offsets,self.point_force_buffer,dim=-1))
            self.position_apply_buffer.copy_(self.asset.data.body_pos_w)

    @observation
    def offline_force_prompt(self):
        # Fixed point ordering, only this frame; no force history or estimator.
        q=self.asset.data.root_quat_w[:,None].expand(-1,len(self.point_names),-1)
        return quat_apply_inverse(q,self.point_force_buffer).flatten(1)/30.

    def offline_force_prompt_sym(self):
        n=3*len(self.point_names);return SymmetryTransform(torch.arange(n),torch.ones(n))
