"""Motion-observable continuous surface contacts with direct Cartesian springs."""
import torch
from .surface_compliance import SurfaceComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply
from active_adaptation.utils.body_guidance import limit_force
from active_adaptation.utils.contact_observability import greedy_contacts
from active_adaptation.utils.symmetry import SymmetryTransform

class ObservableSurfaceCommand(SurfaceComplianceCommand):
    def __init__(self,env,*args,pool_size=64,spring_k=80.,contact_period=.8,upper_body=False,covariance_path=None,full_span=False,coverage_prior=0.,geometry_path="artifacts/ceer_segmented_jacobian/geometry.json",**kwargs):
        # Own-state latent memory is a public observation, not a motion label.
        from .sonic_compliance import SonicComplianceCommand
        super().__init__(env,*args,**kwargs)
        self.pool_size=int(pool_size);self.spring_k=float(spring_k);self.contact_period=float(contact_period)
        import json
        from active_adaptation.utils.surface_fk import SurfaceFK
        from pathlib import Path
        geometry=json.loads(Path(geometry_path).read_text())
        self.upper_body=bool(upper_body);self.full_span=bool(full_span);self.coverage_prior=float(coverage_prior)
        self.coverage_bodies=torch.tensor([self.asset.body_names.index(b) for b in ["pelvis","left_wrist_yaw_link","right_wrist_yaw_link"]],device=self.device)
        self.covariance=None
        if covariance_path:
            meta=json.loads(Path(covariance_path).read_text())
            indices=list(range(6))+[6+meta['joint_names'].index(n) for n in self.asset.joint_names]
            C=torch.tensor(meta['matrix'],device=self.device,dtype=torch.float32)
            self.covariance=C[indices][:,indices]
        self.fk=SurfaceFK(geometry,self.dataset.joint_names,self.device)
        self.motion_body_map=torch.tensor([self.fk.bodies.index(b) for b in self.asset.body_names],device=self.device)

    def _reference(self):
        super()._reference()
        self.next_motion=self.dataset.get_slice(None,torch.minimum(self.t+1,self.lengths-1),1)

    def _select(self,ids):
        if not len(ids):return
        self.force[ids]=0;self.desired[ids]=0
        for ids in ids.split(self.selection_chunk):
            n=len(ids);body,tri,bary=self.surface.sample((n,self.pool_size))
            _,pos,jac=self._geometry(ids,body,tri,bary)
            valid=pos[...,2]>=self.env.scene.env_origins[ids,None,2]+.06
            if self.upper_body:
                allowed=torch.tensor([not any(s in b for s in ('hip','knee','ankle')) for b in self.asset.body_names],device=self.device)
                valid &= allowed[body]
            jac=jac*valid[...,None,None]
            deltas=[]
            for h in ([0,10,20,30] if self.full_span else [0,4,8]):
                ref=self.dataset.get_slice(ids,self.t[ids]+h,2)
                delta=torch.zeros(n,6+self.asset.num_joints,device=self.device)
                delta[:,:3]=ref.root_pos_w[:,1]-ref.root_pos_w[:,0]
                # World-frame small rotation vector, consistent with geometric J.
                from active_adaptation.utils.math import quat_mul,quat_conjugate
                rel=quat_mul(ref.root_quat_w[:,1],quat_conjugate(ref.root_quat_w[:,0]))
                rel=rel*torch.where(rel[:,:1]<0,-1.,1.)
                v=rel[:,1:];angle=2*torch.atan2(v.norm(dim=-1,keepdim=True),rel[:,:1])
                delta[:,3:6]=v*angle/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                delta[:,6+self.all_joint_idx_asset]=ref.joint_pos[:,1,self.all_joint_idx_dataset]-ref.joint_pos[:,0,self.all_joint_idx_dataset]
                deltas.append(delta)
            deltas=torch.stack(deltas,1)
            if self.covariance is not None:
                from isaaclab.utils.math import matrix_from_quat
                R=matrix_from_quat(self.asset.data.root_quat_w[ids])
                jac[...,:3]=jac[...,:3]@R[:,None];jac[...,3:6]=jac[...,3:6]@R[:,None]
                deltas[...,:3]=deltas[...,:3]@R;deltas[...,3:6]=deltas[...,3:6]@R
            selected,active=greedy_contacts(jac,deltas,self.max_contacts,covariance=self.covariance,coverage=body[...,None]==self.coverage_bodies,coverage_penalty=self.coverage_prior)
            row=torch.arange(n,device=self.device)[:,None]
            self.contact_body[ids]=body[row,selected];self.contact_triangle[ids]=tri[row,selected];self.contact_bary[ids]=bary[row,selected]
            self.local_points[ids]=(self.contact_triangle[ids]*self.contact_bary[ids,...,None]).sum(-2)
            self.active[ids]=active & valid[row,selected]
            self.age[ids]=0;self.remaining[ids]=self.contact_period;self.releasing[ids]=False

    def _solve(self):
        row=self.rows[:,None];body=self.contact_body;motion_ids=self.motion_body_map[body]
        local=self.local_points;ref=self.next_motion
        positions,rotations=self.fk(ref.joint_pos[:,0],ref.root_pos_w[:,0],ref.root_quat_w[:,0])
        anchor=positions[row,motion_ids]+quat_apply(rotations[row,motion_ids],local)+self.env.scene.env_origins[:,None]
        current=self.asset.data.body_pos_w[row,body]+quat_apply(self.asset.data.body_quat_w[row,body],local)
        self.desired.copy_(limit_force(self.spring_k*(anchor-current)*self.active[...,None]))
        self.desired[self.releasing]=0
        if not self.guidance_enabled:self.desired.zero_()
        self.contact_error.zero_()

    def before_update(self):
        MotionTrackingCommand.before_update(self);self.update_reward_target();self._reference()
        self.age+=self.env.step_dt;self.remaining-=self.env.step_dt
        # Adaptive unloading plus two-step margin for potentially increasing force.
        self.releasing|=self.remaining<=self.force.norm(dim=-1).amax(-1)/150.+2*self.env.step_dt+1e-6
        ready=(self.remaining<=0)&(self.force.norm(dim=-1).amax(-1)<1e-5)
        self._select(ready.nonzero().squeeze(-1));self._solve();self.log_step+=1
        if self.log_step%16==0:self.env.extra.update({'surface/total_force_mean_N':self.force.norm(dim=-1).sum(-1).mean().item(),'surface/active_contacts':self.active.sum(-1).float().mean().item(),'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.all_joint_idx_asset]-self.qref).abs().mean().item()})

    @observation
    def latent_reference(self):
        m=self.env.action_manager
        return torch.cat((m.latent_residual,m.latent_endpoint,m.last_latent),-1)

    def latent_reference_sym(self):return SymmetryTransform(torch.arange(96),torch.ones(96))
