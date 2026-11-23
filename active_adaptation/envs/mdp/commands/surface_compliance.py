"""1–3 continuous mesh contacts: random or motion-conditioned optimization."""
import torch
from .sonic_compliance import SonicComplianceCommand
from .motion_tracking import MotionTrackingCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply,quat_apply_inverse
from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
from active_adaptation.utils.body_guidance import slew_force
from active_adaptation.utils.surface_guidance import SurfaceSampler,geometry,solve,objective,project_simplex
from active_adaptation.utils.symmetry import SymmetryTransform

class SurfaceComplianceCommand(SonicComplianceCommand):
    def __init__(self,env,*args,surface_path,contact_mode='random',population=12,
                 position_iterations=4,selection_chunk=256,history_offsets=(0,1,2,4,8,16),max_contacts=3,**kwargs):
        super().__init__(env,*args,**kwargs)
        if contact_mode not in ('random','optimized'):raise ValueError(contact_mode)
        self.contact_mode=contact_mode;self.population=int(population);self.position_iterations=int(position_iterations);self.selection_chunk=int(selection_chunk)
        self.surface=SurfaceSampler(surface_path,self.asset.body_names,self.device)
        n=self.num_envs;self.max_contacts=int(max_contacts);k=self.max_contacts
        if k not in (3,5):raise ValueError(k)
        self.slots=torch.zeros(n,k,dtype=torch.long,device=self.device)
        self.active=torch.ones(n,k,dtype=torch.bool,device=self.device)
        self.force=torch.zeros(n,k,3,device=self.device);self.desired=torch.zeros_like(self.force)
        self.contact_body=torch.zeros(n,k,dtype=torch.long,device=self.device)
        self.contact_triangle=torch.zeros(n,k,3,3,device=self.device);self.contact_bary=torch.zeros(n,k,3,device=self.device);self.contact_bary[...,0]=1
        self.local_points=torch.zeros(n,k,3,device=self.device)
        self.age=torch.zeros(n,device=self.device);self.bad_count=torch.zeros(n,dtype=torch.long,device=self.device);self.contact_error=torch.zeros(n,device=self.device)
        self.history_offsets=tuple(history_offsets);self.token_dim=k*(len(self.asset.body_names)+10)
        self.contact_history=torch.zeros(n,max(self.history_offsets)+1,self.token_dim,device=self.device)
        self.fresh=torch.ones(n,dtype=torch.bool,device=self.device);self.history_tick=-1
        self.force_apply_world=True

    def reset(self,ids):
        super().reset(ids)
        self.contact_history[ids]=0;self.fresh[ids]=True

    def _reference(self):
        nxt=self.dataset.get_slice(None,torch.minimum(self.t+1,self.lengths-1),1)
        self.qref=self._motion.joint_pos[:,0,self.all_joint_idx_dataset]
        q1=nxt.joint_pos[:,0,self.all_joint_idx_dataset]
        qactual=self.asset.data.joint_pos[:,self.all_joint_idx_asset]
        self.target=torch.zeros(self.num_envs,6+self.asset.num_joints,device=self.device)
        self.target[:,6+self.all_joint_idx_asset]=80*(q1-self.qref)+20*(self.qref-qactual)
        dp=nxt.root_pos_w[:,0]-self._motion.root_pos_w[:,0]
        root_error=self._motion.root_pos_w[:,0]+self.env.scene.env_origins-self.asset.data.root_pos_w
        fd=60*(dp+root_error)+8*(self._motion.root_lin_vel_w[:,0]-self.asset.data.root_lin_vel_w)
        fd*=40/fd.norm(dim=-1,keepdim=True).clamp_min(40)
        yaw=yaw_from_quat(nxt.root_quat_w[:,0])-yaw_from_quat(self.asset.data.root_quat_w)
        yaw=torch.atan2(yaw.sin(),yaw.cos())
        td=(8*yaw+1.5*(self._motion.root_ang_vel_w[:,0,2]-self.asset.data.root_ang_vel_w[:,2])).clamp(-4,4)
        self.target[:,:3]=fd;self.target[:,5]=td
        self.weights=torch.full_like(self.target,1/.3);self.weights[:,:6]=0
        # Do not force zero resultant for local limb motion. Root terms become
        # active only for commanded root motion or appreciable tracking drift.
        translate=(self._motion.root_lin_vel_w[:,0].norm(dim=-1)>.05)|(root_error.norm(dim=-1)>.05)
        turn=(self._motion.root_ang_vel_w[:,0,2].abs()>.1)|(yaw.abs()>.05)
        self.weights[:,:3]=2*translate[:,None];self.weights[:,5]=(2/.3)*turn
        self.raw_jac=self.asset.root_physx_view.get_jacobians()
        if self.raw_jac.shape[1:]!=(self.asset.num_bodies,6,6+self.asset.num_joints):raise RuntimeError('Floating-base Jacobian required')

    def _geometry(self,ids,body,tri,bary):
        data=self.asset.data
        local,pos,jac=geometry(self.raw_jac[ids],data.body_pos_w[ids],data.body_com_pos_w[ids],data.body_quat_w[ids],body,tri,bary)
        arms=pos-data.root_pos_w[ids,None];x,y,z=arms.unbind(-1);zero=torch.zeros_like(x)
        jac[...,:3]=torch.eye(3,device=self.device)
        jac[...,3:6]=torch.stack((zero,z,-y,-z,zero,x,y,-x,zero),-1).reshape(*arms.shape[:2],3,3)
        return local,pos,jac

    def _sample(self,ids):
        body,tri,bary=self.surface.sample((len(ids),self.max_contacts))
        for _ in range(12):
            _,pos,_=self._geometry(ids,body,tri,bary)
            bad=pos[...,2]<self.env.scene.env_origins[ids,None,2]+.06
            if not bad.any():break
            b,t,u=self.surface.sample((int(bad.sum()),));body[bad]=b;tri[bad]=t;bary[bad]=u
        # All retained samples must be above the floor. Fixed-body torso fallback
        # remains continuously sampled, and is only used if rejection exhausted.
        _,pos,_=self._geometry(ids,body,tri,bary);bad=pos[...,2]<self.env.scene.env_origins[ids,None,2]+.06
        if bad.any():
            valid=(self.surface.body==self.asset.body_names.index('torso_link')).nonzero().squeeze(-1)
            ii=valid[torch.randint(len(valid),(int(bad.sum()),),device=self.device)]
            body[bad]=self.surface.body[ii];tri[bad]=self.surface.triangles[ii]
        return body,tri,bary

    def _select(self,ids):
        if not len(ids):return
        for ids in ids.split(self.selection_chunk):
            n=len(ids);targets=[self.target[ids]];weights=[self.weights[ids]]
            if self.contact_mode=='optimized':
                # A 0.16 s lookahead scores contact sets across multiple future
                # motion increments; actual-state feedback is held at current error.
                for h in [4,8]:
                    ref=self.dataset.get_slice(ids,self.t[ids]+h,2)
                    target=self.target[ids].clone()
                    target[:,6+self.all_joint_idx_asset]=80*(ref.joint_pos[:,1,self.all_joint_idx_dataset]-ref.joint_pos[:,0,self.all_joint_idx_dataset])+20*(self.qref[ids]-self.asset.data.joint_pos[ids][:,self.all_joint_idx_asset])
                    targets.append(target);weights.append(self.weights[ids])
            targets=torch.stack(targets,1);weights=torch.stack(weights,1)
            best=torch.full((n,),float('inf'),device=self.device)
            for candidate in range(self.population if self.contact_mode=='optimized' else 1):
                body,tri,bary=self._sample(ids)
                count=torch.randint(1,self.max_contacts+1,(n,),device=self.device) if self.contact_mode=='random' else torch.full((n,),candidate%self.max_contacts+1,device=self.device)
                active=torch.arange(self.max_contacts,device=self.device)[None]<count[:,None]
                _,pos,jac=self._geometry(ids,body,tri,bary);score=objective(jac,targets,weights,active)
                score+=(pos[...,2]<self.env.scene.env_origins[ids,None,2]+.06).any(-1)*100
                take=score<best
                if candidate==0:
                    chosen_body=body;chosen_tri=tri;chosen_bary=bary;chosen_active=active
                else:
                    chosen_body[take]=body[take];chosen_tri[take]=tri[take];chosen_bary[take]=bary[take];chosen_active[take]=active[take]
                best=torch.minimum(best,score)
            if self.contact_mode=='optimized':
                # Continuous constrained optimization on the selected triangles.
                for _ in range(self.position_iterations):
                    with torch.enable_grad():
                        u=chosen_bary.detach().requires_grad_(True)
                        _,_,jac=self._geometry(ids,chosen_body,chosen_tri,u)
                        loss=objective(jac,targets.detach(),weights.detach(),chosen_active)
                        grad=torch.autograd.grad(loss.sum(),u)[0]
                    proposal=project_simplex(chosen_bary-.15*grad/grad.norm(dim=-1,keepdim=True).clamp_min(1))
                    _,pos,jac=self._geometry(ids,chosen_body,chosen_tri,proposal)
                    score=objective(jac,targets,weights,chosen_active)
                    take=(score<best)&(pos[...,2]>=self.env.scene.env_origins[ids,None,2]+.06).all(-1)
                    chosen_bary[take]=proposal[take];best=torch.where(take,score,best)
            self.contact_body[ids]=chosen_body;self.contact_triangle[ids]=chosen_tri;self.contact_bary[ids]=chosen_bary;self.active[ids]=chosen_active
            self.local_points[ids]=(chosen_tri*chosen_bary[...,None]).sum(-2)
            self.remaining[ids]=1+2*torch.rand(n,device=self.device);self.releasing[ids]=False;self.age[ids]=0;self.bad_count[ids]=0

    def _solve(self):
        _,_,jac=self._geometry(self.rows,self.contact_body,self.contact_triangle,self.contact_bary)
        self.desired.copy_(solve(jac,self.target,self.weights,self.active,self.force))
        self.desired[self.releasing]=0
        if not self.guidance_enabled:self.desired.zero_()
        achieved=torch.einsum('nkcd,nkc->nd',jac,self.desired)
        self.contact_error=((achieved-self.target)*self.weights).norm(dim=-1)/(self.target*self.weights).norm(dim=-1).clamp_min(.1)

    def before_update(self):
        MotionTrackingCommand.before_update(self);self.update_reward_target();self._reference()
        self.remaining-=self.env.step_dt;self.age+=self.env.step_dt
        self.bad_count=torch.where((self.contact_error>.8)&(self.age>.5),self.bad_count+1,0)
        self.releasing|=(self.remaining<=0)
        if self.contact_mode=='optimized':self.releasing|=self.bad_count>=10
        ready=self.releasing&(self.force.norm(dim=-1).amax(-1)<1e-5)
        self._select(ready.nonzero().squeeze(-1));self._solve();self.log_step+=1
        if self.log_step%16==0:
            self.env.extra.update({'surface/total_force_mean_N':self.force.norm(dim=-1).sum(-1).mean().item(),
              'surface/total_force_max_N':self.force.norm(dim=-1).sum(-1).max().item(),
              'surface/active_contacts':(self.force.norm(dim=-1)>1e-5).sum(-1).float().mean().item(),
              'surface/relative_fit_error':self.contact_error.mean().item(),
              'surface/releasing_fraction':self.releasing.float().mean().item(),
              'tracking/joint_mae_rad':(self.asset.data.joint_pos[:,self.all_joint_idx_asset]-self.qref).abs().mean().item()})

    def step(self,substep):
        self.force.copy_(slew_force(self.force,self.desired,self.env.physics_dt))
        self.force_apply_buffer.zero_();self.torque_apply_buffer.zero_()
        offset=quat_apply(self.asset.data.body_quat_w[self.rows[:,None],self.contact_body],self.local_points)
        ids=self.contact_body[:,:,None].expand(-1,-1,3)
        self.force_apply_buffer.scatter_add_(1,ids,self.force)
        self.torque_apply_buffer.scatter_add_(1,ids,torch.cross(offset,self.force,dim=-1))
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w);self.asset.has_external_wrench=False

    @observation
    def surface_contacts(self):
        quat=self.asset.data.root_quat_w[:,None].expand(-1,self.max_contacts,-1)
        pos=self.asset.data.body_pos_w[self.rows[:,None],self.contact_body]+quat_apply(self.asset.data.body_quat_w[self.rows[:,None],self.contact_body],self.local_points)
        pos=quat_apply_inverse(quat,pos-self.asset.data.root_pos_w[:,None])
        f=quat_apply_inverse(quat,self.force)/30
        onehot=torch.nn.functional.one_hot(self.contact_body,len(self.asset.body_names)).float()
        # Reveal only contacts whose force has actually been applied. Planned
        # contact locations must not leak next-motion information at zero force.
        visible=self.active & (self.force.norm(dim=-1)>1e-5)
        token=torch.cat([onehot,self.local_points,pos,f,visible[...,None].float()],-1)*visible[...,None]
        # Exact canonical lexicographic order is invariant to contact slot order.
        order=torch.arange(self.max_contacts,device=self.device).expand(self.num_envs,-1)
        keys=torch.cat([visible[...,None].float(),self.contact_body[...,None].float(),self.local_points,f],-1)*visible[...,None]
        for axis in reversed(range(keys.shape[-1])):
            value=keys.gather(1,order[...,None].expand(-1,-1,keys.shape[-1]))[...,axis]
            order=order.gather(1,value.argsort(dim=-1,stable=True))
        current=token.gather(1,order[...,None].expand_as(token)).flatten(1)
        if self.history_tick!=self.log_step:
            self.contact_history[:,1:]=self.contact_history[:,:-1].clone();self.contact_history[:,0]=current;self.history_tick=self.log_step
        self.contact_history[self.fresh]=0;self.contact_history[self.fresh,0]=current[self.fresh];self.fresh.zero_()
        return self.contact_history[:,self.history_offsets].flatten(1)

    def surface_contacts_sym(self):
        n=len(self.history_offsets)*self.token_dim;return SymmetryTransform(torch.arange(n),torch.ones(n))
