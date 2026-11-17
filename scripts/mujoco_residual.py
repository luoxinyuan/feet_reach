"""Portable causal adapters for frozen low-level trackers and residual students.

No Isaac, dataset, training module, reference motion, or joint predictor imports.
"""
from pathlib import Path
import hashlib
import json
import os
import urllib.request
import mujoco
import numpy as np
import torch
from mujoco_whole_body_student import Simulation


def sonic_models(bundle):
    """Pinned public ONNX exports; validate before loading, download atomically."""
    meta=json.loads((bundle/'metadata.json').read_text())
    cache=Path(__file__).resolve().parents[1]/'.cache/mujoco-sonic'
    paths=[]
    for name,spec in meta['onnx_resources'].items():
        path=bundle/name
        if not path.exists():path=cache/name
        if not path.exists():
            cache.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.part.'+str(os.getpid()))
            print('Downloading official SONIC '+name+' (first run only)...',flush=True)
            try:
                urllib.request.urlretrieve(spec['url'],temporary)
                digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
                if digest!=spec['sha256']:raise RuntimeError('SONIC download checksum mismatch: '+name)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise RuntimeError('SONIC checksum mismatch; remove and redownload: '+str(path))
        paths.append(path)
    return dict(zip(meta['onnx_resources'],paths))


def quat_mul(a,b):
    out=np.zeros(4);mujoco.mju_mulQuat(out,np.asarray(a,dtype=float),np.asarray(b,dtype=float));return out


def quat_matrix(q):
    out=np.zeros(9);mujoco.mju_quat2Mat(out,np.asarray(q,dtype=float));return out.reshape(3,3)


def wrist_interface(q,edges):
    states={'pelvis':(np.zeros(3),np.array([1.,0,0,0]))}
    for parent,child,index,p0,r0,p1,r1,axis in edges:
        pos,rot=states[parent]
        joint=np.array([1.,0,0,0]) if index is None else np.r_[np.cos(q[index]/2),np.sin(q[index]/2)*np.array(axis)]
        child_rot=quat_mul(quat_mul(quat_mul(rot,r0),joint),np.array(r1)*[1,-1,-1,-1])
        child_pos=pos+quat_matrix(rot)@p0-quat_matrix(child_rot)@p1
        states[child]=(child_pos,child_rot)
    positions=[];rotations=[]
    for side in ('left','right'):
        pos,rot=states[side+'_hand_mimic'];positions.extend(pos)
        # Same shortest axis-angle convention as Isaac's axis_angle_from_quat.
        if rot[0]<0:rot=-rot
        sine=np.linalg.norm(rot[1:]);angle=2*np.arctan2(sine,rot[0])
        rotations.extend(rot[1:]*(2. if sine<1e-8 else angle/sine))
    return np.array(positions+rotations,dtype=np.float32)


class ResidualSimulation(Simulation):
    def __init__(self,bundle,model_path,physics_dt=None):
        super().__init__(bundle,model_path,physics_dt)
        self.backend=self.meta['backend']
        if not self.meta.get('self_collisions', True):
            # Robot-ground contacts remain enabled; robot-robot pairs are disabled.
            colliders=(self.model.geom_bodyid>0)&((self.model.geom_contype!=0)|(self.model.geom_conaffinity!=0))
            self.model.geom_contype[colliders]=2
            self.model.geom_conaffinity[colliders]=1
            mujoco.mj_forward(self.model,self.data)
        if self.backend in ('sonic','sonic-force-history'):
            import onnxruntime as ort
            paths=sonic_models(bundle)
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            self.encoder=ort.InferenceSession(str(paths['model_encoder.onnx']),sess_options=options,providers=['CPUExecutionProvider'])
            self.decoder=ort.InferenceSession(str(paths['model_decoder.onnx']),sess_options=options,providers=['CPUExecutionProvider'])
        else:
            self.low_policy=torch.jit.load(str(bundle/'low_level.pt'),map_location='cpu').eval()
        self.last_low_obs=None

    def reset(self):
        super().reset()
        self.reference=np.clip(self.data.qpos[self.qadr],self.meta['joint_lower'],self.meta['joint_upper']).copy()
        self.reference_velocity=np.zeros(29)
        self.low_action=np.zeros(29);self.low_actions=np.zeros((3,29));self.low_applied=np.zeros(29)
        self.low_histories=[np.zeros((10,d)) for d in (3,29,29,29,3)]
        self.low_fresh=True;self.age=0
        self.rng=np.random.default_rng(42)
        lowcfg=self.meta.get('low_action_config',{})
        maxdelay=lowcfg.get('max_delay',0)
        lo,hi=maxdelay if isinstance(maxdelay,list) else (0,maxdelay)
        self.delay=int(self.rng.integers(lo,hi+1));self.boot_delay=self.delay if lowcfg.get('boot_protect',False) else 0
        self.alpha=float(self.rng.uniform(*lowcfg.get('alpha',[.9,.9])))
        self.boot_q=self.data.qpos[self.qadr].copy()
        self.last_obs=np.zeros(self.meta['obs_dim'],dtype=np.float32)

    def observation(self):
        public=super().observation()
        q=self.data.qpos[self.qadr]
        proprioception=np.r_[public[:238],self.data.qvel[self.vadr],self.reference,
                             self.reference-q,self.reference_velocity].astype(np.float32)
        # DenseForcePPO's teacher actor is built with in_keys
        # ['policy', 'force_input_'], so its normalized 354-D proprioception
        # comes first and the 9-D applied force is appended at the end.  Keep
        # the CEER teacher input layout identical to the trained TensorDict
        # concatenation. Other backends retain their established layout.
        if self.backend=='ceer':
            return np.r_[proprioception,public[238:]].astype(np.float32)
        return np.r_[public,self.data.qvel[self.vadr],self.reference,
                     self.reference-q,self.reference_velocity].astype(np.float32)

    def integrate_reference(self,raw):
        q=self.data.qpos[self.qadr];dt=self.meta['policy_dt']
        requested=dt*self.meta['reference_speed']*np.tanh(raw)
        lower=np.maximum(self.meta['joint_lower'],q-self.meta['max_reference_error'])
        upper=np.minimum(self.meta['joint_upper'],q+self.meta['max_reference_error'])
        proposal=np.clip(self.reference+requested,lower,upper)
        delta=proposal-self.reference
        delta=np.where(delta*requested>0,np.sign(delta)*np.minimum(np.abs(delta),np.abs(requested)),0.)
        self.reference+=delta;self.reference_velocity=delta/dt

    def sonic_inputs(self):
        rot=self.data.xmat[self.root].reshape(3,3)
        yaw=np.arctan2(rot[1,0],rot[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        upright=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        orientation=(rot.T@upright)[:,:2].ravel()
        velocity=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,velocity,1)
        values=(velocity[:3],self.data.qpos[self.qadr]-self.q0,self.data.qvel[self.vadr],self.low_action,rot.T@np.array([0.,0.,-1.]))
        for hist,value in zip(self.low_histories,values):
            hist[:-1]=hist[1:].copy();hist[-1]=value
            if self.low_fresh:hist[:]=value
        self.low_fresh=False
        prop=np.concatenate([h.ravel() for h in self.low_histories]).astype(np.float32)[None]
        encoder=np.zeros((1,1762),dtype=np.float32)
        encoder[:,4:294]=np.tile(self.reference,10)
        encoder[:,601:661]=np.tile(orientation,10)
        return encoder,prop

    def ceer_inputs(self):
        rot=self.data.xmat[self.root].reshape(3,3)
        # MuJoCo free-joint qvel is ordered [angular, linear], whereas the
        # Isaac asset API exposes root_lin_vel_b directly.  Use the body
        # spatial velocity here so the CEER command receives the same local
        # translational velocity in both simulators.
        root_spatial_velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                                 self.root,root_spatial_velocity,1)
        root_velocity=root_spatial_velocity[3:]
        command=np.r_[self.data.qpos[2],root_velocity[:2],1.,0.,self.meta['ceer_force_limit']]
        public=super().observation()
        wrists=wrist_interface(self.reference,self.meta['fk_edges'])
        obs=np.r_[max(0.,1-self.age/25),command,wrists,public[:151],self.low_actions.ravel()].astype(np.float32)[None]
        reference=np.tile(self.reference[self.meta['target_order']],(1,5,1)).astype(np.float32)
        assert obs.shape==(1,257)
        return obs,reference

    def _update_policy(self):
        self.last_obs=self.observation()
        with torch.inference_mode():
            obs=(torch.from_numpy(self.last_obs)-self.loc)/self.scale
            raw=self.policy(obs[None])[0].numpy()
        if not np.isfinite(raw).all():raise RuntimeError('Non-finite residual action')
        self.last_action=raw.copy();self.actions[1:]=self.actions[:-1].copy();self.actions[0]=raw
        self.integrate_reference(raw)
        if self.backend in ('sonic','sonic-force-history'):
            encoder,prop=self.sonic_inputs();token=self.encoder.run(None,{self.encoder.get_inputs()[0].name:encoder})[0]
            inp=np.concatenate((token,prop),axis=-1)
            self.low_action=self.decoder.run(None,{self.decoder.get_inputs()[0].name:inp})[0][0]
            self.last_low_obs=(encoder.copy(),prop.copy())
            self.target=self.q0+np.array(self.meta['action_scale'])*self.low_action
        else:
            obs,reference=self.ceer_inputs();self.last_low_obs=(obs.copy(),reference.copy())
            with torch.inference_mode():self.low_action=self.low_policy(torch.from_numpy(obs),torch.from_numpy(reference))[0].numpy()
            self.low_actions[1:]=self.low_actions[:-1].copy();self.low_actions[0]=self.low_action.clip(-10,10)
            cfg=self.meta['low_action_config'];jitter=cfg.get('alpha_jit_scale')
            if jitter is not None:self.alpha=float(np.clip(self.alpha+self.rng.uniform(-jitter,jitter),*cfg['alpha_wide']))
        if not np.isfinite(self.low_action).all():raise RuntimeError('Non-finite low-level action')
        self.age+=1

    def _before_torque(self):
        if self.backend in ('ceer','ceer-force-history','jacobian','segmented','dense-estimator'):
            substep=self.steps%self.decimation
            idx=(self.delay-substep+self.decimation-1)//self.decimation
            delayed=self.low_actions[idx]
            self.low_applied+=self.alpha*(delayed-self.low_applied)
            self.target=self.q0+np.array(self.meta['action_scale'])*self.low_applied
            if self.boot_delay>0:self.target=self.boot_q.copy()
            self.boot_delay=max(0,self.boot_delay-1)


class DenseResidualSimulation(ResidualSimulation):
    """23-point policy: infer per-link forces from causal proprioceptive history."""
    def __init__(self,bundle,model_path,physics_dt=None):
        self.force_estimator=torch.jit.load(str(bundle/'estimator.pt'),map_location='cpu').eval()
        super().__init__(bundle,model_path,physics_dt)

    def reset(self):
        super().reset()
        self.estimator_history=np.zeros((10,93),dtype=np.float32)
        self.history_fresh=True
        self.estimated_forces=np.zeros((len(self.points),3),dtype=np.float32)

    def observation(self):
        # The dense actor was trained with separate `policy` and `force_input_`
        # keys. Its policy observation intentionally excludes privileged force_gt.
        full=super().observation()
        base=np.concatenate((full[:238],full[238+self.forces.size:])).astype(np.float32)
        rot=self.data.xmat[self.root].reshape(3,3)
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,velocity,1)
        current=np.concatenate((velocity[:3]/5.,rot.T@np.array([0.,0.,-1.]),
            self.data.qpos[self.qadr],self.data.qvel[self.vadr]/10.,
            np.clip(self.low_action,-10,10)/5.)).astype(np.float32)
        if self.history_fresh:
            self.estimator_history[:]=current
            self.history_fresh=False
        else:
            self.estimator_history[1:]=self.estimator_history[:-1].copy()
            self.estimator_history[0]=current
        with torch.inference_mode():
            estimate=self.force_estimator(torch.from_numpy(self.estimator_history.reshape(1,-1)))[0].numpy()
        if not np.isfinite(estimate).all():
            raise RuntimeError('Non-finite dense force estimate')
        self.estimated_forces=estimate.reshape(len(self.points),3).copy()
        return np.concatenate((base,estimate)).astype(np.float32)


class ForceHistorySimulation(ResidualSimulation):
    """Three-point teacher with the six sampled force prompts used in training."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='ceer-force-history':
            raise ValueError('ForceHistorySimulation requires the ceer-force-history bundle.')
        self.force_history_offsets=tuple(int(x) for x in meta['force_history_offsets'])
        if self.force_history_offsets!=(0,1,2,4,8,16):
            raise ValueError(f'Unexpected force-history offsets: {self.force_history_offsets}')
        self.force_history=np.zeros((max(self.force_history_offsets)+1,9),dtype=np.float32)
        self.force_history_tick=-1
        super().__init__(bundle,model_path,physics_dt)

    def reset(self):
        super().reset()
        self.force_history[:]=0.
        self.force_history_tick=-1

    def observation(self):
        # The teacher's TensorDict input order is the 354-D proprioception first,
        # followed by six nine-dimensional applied-force prompts. Forces are
        # root-local and normalized by 30 N, matching dense_force_gt_history().
        public=Simulation.observation(self)
        q=self.data.qpos[self.qadr]
        proprioception=np.r_[public[:238],self.data.qvel[self.vadr],self.reference,
                             self.reference-q,self.reference_velocity].astype(np.float32)
        if proprioception.size!=self.meta['base_obs_dim']:
            raise RuntimeError(f'Expected {self.meta["base_obs_dim"]}-D proprioception, got {proprioception.size}')
        if self.force_history_tick!=self.steps:
            self.force_history[1:]=self.force_history[:-1].copy()
            rot=self.data.xmat[self.root].reshape(3,3)
            self.force_history[0]=(self.forces@rot).reshape(-1)/self.meta['force_limit']
            self.force_history_tick=self.steps
        sampled=self.force_history[list(self.force_history_offsets)].reshape(-1)
        result=np.r_[proprioception,sampled].astype(np.float32)
        if result.size!=self.meta['obs_dim']:
            raise RuntimeError(f'Expected {self.meta["obs_dim"]}-D teacher input, got {result.size}')
        return result


class SonicForceHistorySimulation(ResidualSimulation):
    """SONIC low-level controller with the trained causal force prompt."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='sonic-force-history':
            raise ValueError('SonicForceHistorySimulation requires the sonic-force-history bundle.')
        self.force_history_offsets=tuple(int(x) for x in meta['force_history_offsets'])
        if self.force_history_offsets!=(0,1,2,4,8,16):
            raise ValueError(f'Unexpected force-history offsets: {self.force_history_offsets}')
        force_dim=3*len(meta['force_points'])
        if force_dim < 3:
            raise ValueError('SONIC force-history bundle must define at least one 3-D force point.')
        if meta['obs_dim'] != meta['base_obs_dim'] + len(self.force_history_offsets)*force_dim:
            raise ValueError('SONIC actor dimension does not match its force-point/history metadata.')
        self.force_history=np.zeros((max(self.force_history_offsets)+1,force_dim),dtype=np.float32)
        self.force_history_tick=-1
        super().__init__(bundle,model_path,physics_dt)

    def reset(self):
        super().reset()
        self.force_history[:]=0.
        self.force_history_tick=-1

    def observation(self):
        # Match the trained task's proprioception, then append the root-local
        # force history used by the teacher actor.
        rot=self.data.xmat[self.root].reshape(3,3)
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                                 self.root,velocity,1)
        policy_obs=np.r_[velocity[:3],rot.T@np.array([0.,0.,-1.]),
                         self.history.ravel(),self.actions.ravel(),
                         self.data.qvel[self.vadr],self.reference,
                         self.reference-self.data.qpos[self.qadr],self.reference_velocity].astype(np.float32)
        if policy_obs.size!=self.meta['base_obs_dim']:
            raise RuntimeError(f'Expected {self.meta["base_obs_dim"]}-D policy observation, got {policy_obs.size}')
        if self.force_history_tick!=self.steps:
            self.force_history[1:]=self.force_history[:-1].copy()
            self.force_history[0]=(self.forces@rot).reshape(-1)/self.meta['force_limit']
            self.force_history_tick=self.steps
        sampled=self.force_history[list(self.force_history_offsets)].reshape(-1)
        result=np.r_[policy_obs,sampled].astype(np.float32)
        if result.size!=self.meta['obs_dim']:
            raise RuntimeError(f'Expected {self.meta["obs_dim"]}-D actor input, got {result.size}')
        return result


class SuperMimicForceHistorySimulation(ResidualSimulation):
    """SuperMimic low-level tracker with a force-history residual."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='supermimic-force-history':
            raise ValueError('SuperMimicForceHistorySimulation requires the supermimic-force-history bundle.')
        self.supermimic_joint_names=list(meta['supermimic_joint_names'])
        self.force_history_offsets=tuple(int(x) for x in meta['force_history_offsets'])
        if self.force_history_offsets!=(0,1,2,4,8,16):
            raise ValueError(f'Unexpected force-history sampling offsets: {self.force_history_offsets}')
        force_dim=3*len(meta['force_points'])
        if force_dim < 3:
            raise ValueError('SuperMimic force-history bundle must define at least one 3-D force point.')
        if meta['obs_dim'] != meta['base_obs_dim'] + len(self.force_history_offsets)*force_dim:
            raise ValueError('SuperMimic actor dimension does not match its force-point/history metadata.')
        self.supermimic_to_model=np.array([meta['joint_names'].index(n) for n in self.supermimic_joint_names])
        if len(self.supermimic_to_model)!=29 or len(set(self.supermimic_to_model.tolist()))!=29:
            raise ValueError('SuperMimic joint order must map one-to-one onto the 29-DOF MuJoCo model.')
        self.supermimic_q0=np.asarray(meta['default_pos'],dtype=np.float32)[self.supermimic_to_model]
        self.supermimic_action_scale=np.asarray(meta['supermimic_action_scale'],dtype=np.float32)
        if self.supermimic_action_scale.shape!=(29,):
            raise ValueError('SuperMimic action scale must have 29 entries.')
        self.force_history=np.zeros((max(self.force_history_offsets)+1,force_dim),dtype=np.float32)
        self.force_history_tick=-1
        super().__init__(bundle,model_path,physics_dt)

    def reset(self):
        super().reset()
        q=self.data.qpos[self.qadr][self.supermimic_to_model]
        self.reference=q.copy()
        self.reference_velocity=np.zeros(29,dtype=np.float32)
        self.reference_origin=q.copy()
        self.low_action=np.zeros(29,dtype=np.float32)
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True
        self.last_force_observation_error=0.
        self.force_history[:]=0.
        self.force_history_tick=-1

    def observation(self):
        # Match the training order: asset-order q history, then SuperMimic
        # action-order qd/reference/error/reference-velocity, then force history.
        public=Simulation.observation(self)
        q=self.data.qpos[self.qadr]
        q_policy=q[self.supermimic_to_model]
        qd_policy=self.data.qvel[self.vadr][self.supermimic_to_model]
        proprioception=np.r_[public[:238],qd_policy,self.reference,
                             self.reference-q_policy,self.reference_velocity].astype(np.float32)
        if proprioception.size!=self.meta['base_obs_dim']:
            raise RuntimeError(f'Expected {self.meta["base_obs_dim"]}-D proprioception, got {proprioception.size}')
        if self.force_history_tick!=self.steps:
            self.force_history[1:]=self.force_history[:-1].copy()
            rot=self.data.xmat[self.root].reshape(3,3)
            self.force_history[0]=(self.forces@rot).reshape(-1)/self.meta['force_limit']
            self.force_history_tick=self.steps
        result=np.r_[proprioception,self.force_history[list(self.force_history_offsets)].reshape(-1)].astype(np.float32)
        if result.size!=self.meta['obs_dim']:
            raise RuntimeError(f'Expected {self.meta["obs_dim"]}-D actor input, got {result.size}')
        return result

    def _update_policy(self):
        self.last_obs=self.observation()
        current_force=(self.forces@self.data.xmat[self.root].reshape(3,3)).reshape(-1)/self.meta['force_limit']
        force_dim=self.forces.size
        self.last_force_observation_error=float(np.max(np.abs(self.last_obs[self.meta['base_obs_dim']:self.meta['base_obs_dim']+force_dim]-current_force)))
        with torch.inference_mode():
            obs=(torch.from_numpy(self.last_obs)-self.loc)/self.scale
            raw=self.policy(obs[None])[0].numpy()
        if not np.isfinite(raw).all():raise RuntimeError('Non-finite residual action')
        raw=np.clip(raw,-10.,10.)
        self.last_action=raw.copy()
        self.actions[1:]=self.actions[:-1].copy();self.actions[0]=raw
        q=self.data.qpos[self.qadr][self.supermimic_to_model]
        requested=self.meta['policy_dt']*self.meta['reference_speed']*np.tanh(raw)
        joint_lower=np.asarray(self.meta['joint_lower'])[self.supermimic_to_model]
        joint_upper=np.asarray(self.meta['joint_upper'])[self.supermimic_to_model]
        lower=np.maximum(joint_lower,q-self.meta['max_reference_error'])
        upper=np.minimum(joint_upper,q+self.meta['max_reference_error'])
        proposal=np.clip(self.reference+requested,lower,upper)
        delta=proposal-self.reference
        delta=np.where(delta*requested>0,np.sign(delta)*np.minimum(np.abs(delta),np.abs(requested)),0.)
        self.reference+=delta
        self.reference_velocity=delta/self.meta['policy_dt']

        root_velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,root_velocity,1)
        rot=self.data.xmat[self.root].reshape(3,3)
        gravity=rot.T@np.array([0.,0.,-1.])
        qd=self.data.qvel[self.vadr][self.supermimic_to_model]
        current_terms=(q-self.supermimic_q0,qd,root_velocity[:3],gravity,self.low_action)
        for history,current in zip(self.supermimic_history,current_terms):
            history[:-1]=history[1:].copy()
            history[-1]=current
            if self.supermimic_history_fresh:history[:]=current
        self.supermimic_history_fresh=False
        low_obs=np.concatenate([x.reshape(-1) for x in self.supermimic_history]).astype(np.float32)[None]

        # The interactive viewer has no motion clip. Use a stationary upright
        # goal at the nominal pose, with the learned residual reference added.
        goal_frame=np.r_[self.reference-self.reference_origin,self.reference_velocity,
                         [0.,0.,-1.],np.zeros(6),self.meta['root_height']].astype(np.float32)
        goal=np.tile(goal_frame,(1,10)).astype(np.float32)
        with torch.inference_mode():
            low=self.low_policy(torch.from_numpy(low_obs),torch.from_numpy(goal))[0].numpy()
        if not np.isfinite(low).all():raise RuntimeError('Non-finite SuperMimic low-level action')
        self.low_action=low.astype(np.float32)
        self.target=self.q0.copy()
        self.target[self.supermimic_to_model]=(
            self.supermimic_q0+self.supermimic_action_scale*self.low_action)
        self.last_low_obs=(low_obs.copy(),goal.copy())
        self.age+=1


class SuperMimicSurfaceLatentSimulation(ResidualSimulation):
    """Observable-surface force actor with a frozen SuperMimic latent decoder."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='supermimic-surface-latent':
            raise ValueError('Expected a supermimic-surface-latent bundle.')
        self.body_names=list(meta['asset_body_names'])
        self.history_offsets=tuple(int(x) for x in meta['force_history_offsets'])
        self.num_contacts=int(meta['max_contacts'])
        self.token_dim=len(self.body_names)+10
        if (self.num_contacts,self.token_dim,self.history_offsets)!=(5,43,(0,1,2,4,8,16)):
            raise ValueError('Unexpected observable-surface token/history layout.')
        if meta['obs_dim']!=372+len(self.history_offsets)*self.num_contacts*self.token_dim:
            raise ValueError('Actor observation dimensions do not match the 5-contact policy.')
        self.body_name_to_isaac={name:i for i,name in enumerate(self.body_names)}
        self.contact_history=np.zeros((max(self.history_offsets)+1,
            self.num_contacts*self.token_dim),dtype=np.float32)
        self.contact_history_tick=-1
        self.selected_body_id=None
        self.selected_point_local=np.zeros(3,dtype=np.float64)
        self.actor_joint_history=np.zeros((5,29),dtype=np.float32)
        self.root_ang_history=np.zeros((1,3),dtype=np.float32)
        self.gravity_history=np.zeros((1,3),dtype=np.float32)
        self.policy_history_fresh=True
        self.action_history=np.zeros((3,32),dtype=np.float32)
        self.latent_residual=np.zeros(32,dtype=np.float32)
        self.latent_endpoint=np.zeros(32,dtype=np.float32)
        self.last_latent=np.zeros(32,dtype=np.float32)
        self.supermimic_joint_names=list(meta['supermimic_joint_names'])
        self.supermimic_to_model=np.asarray([meta['joint_names'].index(n) for n in self.supermimic_joint_names])
        self.supermimic_q0=np.asarray(meta['supermimic_default_pos'],dtype=np.float32)
        self.supermimic_action_scale=np.asarray(meta['supermimic_action_scale'],dtype=np.float32)
        self.latent_scale=float(meta['latent_scale'])
        self.interpolation_frames=int(meta['interpolation_frames'])
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True
        super().__init__(bundle,model_path,physics_dt)
        if len(self.supermimic_to_model)!=29 or len(set(self.supermimic_to_model.tolist()))!=29:
            raise ValueError('SuperMimic joint names must map one-to-one to the MuJoCo model.')
        self._upper_body_ids={self.model.body(n).id for n in self.body_names
            if not any(part in n for part in ('hip','knee','ankle'))}
        self.meta['force_budget']=60.

    def reset(self):
        super().reset()
        specs=self.meta['force_points']
        self.point_names=[p['name'] for p in specs]
        self.points=np.asarray([self.model.body(p['body']).id for p in specs],dtype=np.int32)
        self.point_offsets=np.asarray([p['offset'] for p in specs],dtype=np.float64)
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        self.reference=q.copy()
        self.reference_origin=q.copy()
        self.reference_velocity=np.zeros(29,dtype=np.float32)
        self.low_action=np.zeros(29,dtype=np.float32)
        self.contact_history.fill(0);self.contact_history_tick=-1
        self.actor_joint_history[:]=q[self._asset_joint_to_model()]-self.q0[self._asset_joint_to_model()]
        self.root_ang_history[:]=self._root_ang_vel_body()
        self.gravity_history[:]=self._projected_gravity()
        self.policy_history_fresh=True
        self.action_history.fill(0);self.latent_residual.fill(0)
        self.latent_endpoint.fill(0);self.last_latent.fill(0)
        self.actions=np.zeros((3,32),dtype=np.float32)
        self.last_action=np.zeros(32,dtype=np.float32)
        self.selected_body_id=None
        self.selected_point_local=np.zeros(3,dtype=np.float64)
        self.forces.fill(0)
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True

    def _asset_joint_to_model(self):
        return np.asarray([self.meta['joint_names'].index(n) for n in self.meta['asset_joint_names']])

    def _root_ang_vel_body(self):
        v=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,v,1)
        return v[:3].astype(np.float32)

    def _root_ang_vel_world(self):
        v=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,v,0)
        return v[:3].astype(np.float32)

    def _projected_gravity(self):
        rot=self.data.xmat[self.root].reshape(3,3)
        return (rot.T@np.array([0.,0.,-1.])).astype(np.float32)

    def _after_force_slew(self):
        total=float(np.linalg.norm(self.forces,axis=1).sum())
        budget=float(self.meta['force_budget'])
        if total>budget:self.forces*=budget/total

    def nearest_point(self,body,hit):
        if body not in self._upper_body_ids:return None
        self.selected_body_id=int(body)
        rot=self.data.xmat[body].reshape(3,3)
        self.selected_point_local=rot.T@(np.asarray(hit)-self.data.xpos[body])
        self.points[:]=self.model.body('pelvis').id
        self.point_offsets[:]=0.
        self.points[0]=body
        self.point_offsets[0]=self.selected_point_local
        name=self.model.body(body).name
        self.point_names[0]='surface:'+name
        return 0

    def _contact_frame(self):
        frame=np.zeros((self.num_contacts,self.token_dim),dtype=np.float32)
        rot=self.data.xmat[self.root].reshape(3,3)
        for slot,(body,force,local) in enumerate(zip(self.points,self.forces,self.point_offsets)):
            if np.linalg.norm(force)<=1e-5:continue
            name=self.model.body(int(body)).name
            if name not in self.body_name_to_isaac:continue
            idx=self.body_name_to_isaac[name]
            point=self.data.xpos[body]+self.data.xmat[body].reshape(3,3)@local
            token=frame[slot]
            token[idx]=1.
            token[len(self.body_names):len(self.body_names)+3]=local
            token[len(self.body_names)+3:len(self.body_names)+6]=rot.T@(point-self.data.xpos[self.root])
            token[len(self.body_names)+6:len(self.body_names)+9]=rot.T@force/30.
            token[-1]=1.
        # Training sorts all contacts canonically; inactive slots sort first.
        def sort_key(token):
            visible=token[-1]>0.5
            if not visible:return (0.,0.,0.,0.,0.,0.,0.)
            body=int(np.argmax(token[:len(self.body_names)]))
            return (1.,float(body),*map(float,token[len(self.body_names):len(self.body_names)+6]),
                    *map(float,token[len(self.body_names)+6:len(self.body_names)+9]))
        frame=frame[np.asarray(sorted(range(self.num_contacts),key=lambda i:sort_key(frame[i])))]
        return frame.reshape(-1)

    def _surface_history(self):
        if self.contact_history_tick!=self.steps:
            self.contact_history[1:]=self.contact_history[:-1].copy()
            self.contact_history[0]=self._contact_frame()
            if self.policy_history_fresh:
                self.contact_history[1:]=0.
            self.contact_history_tick=self.steps
        return self.contact_history[np.asarray(self.history_offsets)].reshape(-1)

    def observation(self):
        q=self.data.qpos[self.qadr]
        q_asset=q[self._asset_joint_to_model()]
        q0_asset=self.q0[self._asset_joint_to_model()]
        # `sonic_joint_velocity` in the trained actor is ordered by the
        # SuperMimic action manager's policy joints, while joint_pos_history
        # above intentionally follows the Isaac asset order.  Keep these two
        # observation terms in their distinct training-time joint orders.
        qd=self.data.qvel[self.vadr][self.supermimic_to_model]
        joint_now=(q_asset-q0_asset).astype(np.float32)
        root_ang=self._root_ang_vel_body()
        gravity=self._projected_gravity()
        if self.policy_history_fresh:
            self.actor_joint_history[:]=joint_now
            self.root_ang_history[:]=root_ang
            self.gravity_history[:]=gravity
        else:
            self.actor_joint_history[1:]=self.actor_joint_history[:-1].copy()
            self.actor_joint_history[0]=joint_now
            self.root_ang_history[0]=root_ang
            self.gravity_history[0]=gravity
        policy=np.concatenate((self.root_ang_history.reshape(-1),self.gravity_history.reshape(-1),
            self.actor_joint_history.reshape(-1),self.action_history.reshape(-1),qd.astype(np.float32),
            self.latent_residual,self.latent_endpoint,self.last_latent)).astype(np.float32)
        force_hist=self._surface_history()
        self.policy_history_fresh=False
        result=np.r_[policy,force_hist].astype(np.float32)
        if result.size!=self.meta['obs_dim']:
            raise RuntimeError(f'Expected {self.meta["obs_dim"]}-D actor input; got {result.size}')
        return result

    def _reference_goal(self, stationary=False):
        root_v=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,root_v,0)
        # MuJoCo root body yaw is obtained from its world rotation matrix.
        rot=self.data.xmat[self.root].reshape(3,3)
        heading=np.arctan2(rot[1,0],rot[0,0])
        c,s=np.cos(heading),np.sin(heading)
        world_to_heading=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        linear=np.zeros(3) if stationary else world_to_heading@root_v[3:]
        # The reference-free Isaac goal intentionally exposes only world yaw
        # rate; roll/pitch rates are zeroed.  Feeding all three measured rates
        # here changes the encoded base latent as soon as the body starts to
        # move, creating an unintended feedback path in MuJoCo.
        yaw_rate=0. if stationary else self._root_ang_vel_world()[2]
        angular=np.array([0.,0.,yaw_rate],dtype=np.float32)
        frame=np.r_[self.reference-self.supermimic_q0,self.reference_velocity,
            [0.,0.,-1.],angular,linear,self.data.qpos[2]].astype(np.float32)
        return np.tile(frame,(10,1)).reshape(1,-1)

    def _low_observation(self):
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        qd=self.data.qvel[self.vadr][self.supermimic_to_model].astype(np.float32)
        terms=(q-self.supermimic_q0,qd,self._root_ang_vel_body(),self._projected_gravity(),self.low_action)
        if self.supermimic_history_fresh:
            histories=[]
            for value in terms:histories.append(np.tile(value,(10,1)))
            self.supermimic_history=histories
            self.supermimic_history_fresh=False
        else:
            for history,current in zip(self.supermimic_history,terms):
                history[:-1]=history[1:].copy();history[-1]=current
        return np.concatenate([x.reshape(-1) for x in self.supermimic_history]).astype(np.float32)[None]

    def _update_policy(self):
        self.last_obs=self.observation()
        with torch.inference_mode():
            actor_obs=(torch.from_numpy(self.last_obs)-self.loc)/self.scale
            raw=self.policy(actor_obs[None])[0].numpy().clip(-10.,10.)
        if not np.isfinite(raw).all():raise RuntimeError('Non-finite surface-latent actor output.')
        # This checkpoint's Isaac teacher has appreciable nonzero residuals
        # on an all-zero force prompt (that state is rare during training).
        # The MuJoCo viewer is an interactive compliance tool, so make its
        # intended no-load hold explicit and fade the residual through the
        # existing latent interpolation instead of letting an OOD no-force
        # action make the standing robot twitch.
        force_active=np.linalg.norm(self.forces,axis=-1).max()>1e-5
        stationary=bool(self.meta.get('zero_force_residual_gate',False) and not force_active)
        if stationary:
            raw.fill(0.)
        self.action_history[1:]=self.action_history[:-1].copy();self.action_history[0]=raw
        self.actions[1:]=self.actions[:-1].copy();self.actions[0]=raw
        self.last_action=raw.copy()
        self.latent_endpoint=self.latent_scale*np.tanh(raw)
        self.latent_residual+=(self.latent_endpoint-self.latent_residual)/self.interpolation_frames
        obs=torch.from_numpy(self._low_observation())
        goal=torch.from_numpy(self._reference_goal(stationary=stationary))
        residual=torch.from_numpy(self.latent_residual[None])
        with torch.inference_mode():
            low=self.low_policy(obs,goal,residual)[0].numpy()
        if not np.isfinite(low).all():raise RuntimeError('Non-finite SuperMimic low-level output.')
        self.low_action=low.astype(np.float32)
        self.last_latent=np.clip(self.low_policy_base_latent(goal.numpy())+self.latent_residual,-1.,1.).astype(np.float32)
        target=self.q0.copy()
        target[self.supermimic_to_model]=self.supermimic_q0+self.supermimic_action_scale*self.low_action
        self.target=target
        self.last_low_obs=(obs.numpy().copy(),goal.numpy().copy())

    def low_policy_base_latent(self,goal):
        # The exported wrapper exposes the same encoder; the policy input term
        # needs the current decoded latent for the next actor observation.
        return self.low_policy.base_latent(torch.from_numpy(goal)).numpy()[0]


class SuperMimicSurfaceDirectLatentSimulation(ResidualSimulation):
    """Three-contact surface policy whose 32-D action is the decoder latent."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='supermimic-surface-direct-latent':
            raise ValueError('Expected a direct-latent surface policy bundle.')
        self.body_names=list(meta['asset_body_names'])
        self.history_offsets=tuple(int(x) for x in meta['force_history_offsets'])
        self.num_contacts=int(meta['max_contacts'])
        self.token_dim=len(self.body_names)+10
        if (self.num_contacts,self.token_dim,self.history_offsets)!=(3,43,(0,1,2,4,8,16)):
            raise ValueError('Unexpected 3-contact body-token/history layout.')
        if meta['base_obs_dim']!=363 or meta['obs_dim']!=363+6*3*43:
            raise ValueError('Checkpoint observation must be 363 proprioceptive plus 774 contact-history values.')
        self.body_name_to_isaac={name:i for i,name in enumerate(self.body_names)}
        self.contact_history=np.zeros((17,self.num_contacts*self.token_dim),dtype=np.float32)
        self.contact_history_tick=-1
        self.selected_body_id=None
        self.selected_point_local=np.zeros(3,dtype=np.float64)
        self.actor_joint_history=np.zeros((5,29),dtype=np.float32)
        self.root_ang_history=np.zeros((1,3),dtype=np.float32)
        self.gravity_history=np.zeros((1,3),dtype=np.float32)
        self.policy_history_fresh=True
        self.action_history=np.zeros((3,32),dtype=np.float32)
        self.last_latent=np.zeros(32,dtype=np.float32)
        self.supermimic_joint_names=list(meta['supermimic_joint_names'])
        self.supermimic_to_model=np.asarray([meta['joint_names'].index(n) for n in self.supermimic_joint_names])
        self.supermimic_q0=np.asarray(meta['supermimic_default_pos'],dtype=np.float32)
        self.supermimic_action_scale=np.asarray(meta['supermimic_action_scale'],dtype=np.float32)
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True
        super().__init__(bundle,model_path,physics_dt)
        if len(self.supermimic_to_model)!=29 or len(set(self.supermimic_to_model.tolist()))!=29:
            raise ValueError('SuperMimic joints must map one-to-one onto the MuJoCo model.')
        self._interactive_body_ids={self.model.body(n).id for n in self.body_names}
        self.meta['force_budget']=60.

    def reset(self):
        super().reset()
        self.point_names=[p['name'] for p in self.meta['force_points']]
        self.points=np.asarray([self.model.body(p['body']).id for p in self.meta['force_points']],dtype=np.int32)
        self.point_offsets=np.asarray([p['offset'] for p in self.meta['force_points']],dtype=np.float64)
        self.contact_history.fill(0);self.contact_history_tick=-1
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        q_asset=self.data.qpos[self.qadr][self._asset_joint_to_model()].astype(np.float32)
        self.reference=q.copy();self.reference_velocity=np.zeros(29,dtype=np.float32)
        self.actor_joint_history[:]=q_asset-self.q0[self._asset_joint_to_model()]
        self.root_ang_history[:]=self._root_ang_vel_body();self.gravity_history[:]=self._projected_gravity()
        self.policy_history_fresh=True;self.action_history.fill(0);self.last_latent.fill(0)
        self.actions=np.zeros((3,32),dtype=np.float32);self.last_action=np.zeros(32,dtype=np.float32)
        self.selected_body_id=None;self.selected_point_local.fill(0)
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True;self.forces.fill(0)

    def _asset_joint_to_model(self):
        return np.asarray([self.meta['joint_names'].index(n) for n in self.meta['asset_joint_names']])

    def _root_ang_vel_body(self):
        v=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,v,1)
        return v[:3].astype(np.float32)

    def _projected_gravity(self):
        return (self.data.xmat[self.root].reshape(3,3).T@np.array([0.,0.,-1.])).astype(np.float32)

    def _after_force_slew(self):
        total=float(np.linalg.norm(self.forces,axis=1).sum())
        if total>float(self.meta['force_budget']):self.forces*=float(self.meta['force_budget'])/total

    def nearest_point(self,body,hit):
        if body not in self._interactive_body_ids:return None
        self.selected_body_id=int(body)
        self.selected_point_local=self.data.xmat[body].reshape(3,3).T@(np.asarray(hit)-self.data.xpos[body])
        # The desktop viewer exposes one mouse drag at a time. The trained
        # policy still receives the same three-slot layout used in training.
        self.points[:]=self.model.body('pelvis').id;self.point_offsets[:]=0.
        self.points[0]=body;self.point_offsets[0]=self.selected_point_local
        self.point_names[0]='surface:'+self.model.body(body).name
        return 0

    def _contact_frame(self):
        frame=np.zeros((self.num_contacts,self.token_dim),dtype=np.float32)
        root_rot=self.data.xmat[self.root].reshape(3,3)
        for slot,(body,force,local) in enumerate(zip(self.points,self.forces,self.point_offsets)):
            if np.linalg.norm(force)<=1e-5:continue
            name=self.model.body(int(body)).name
            if name not in self.body_name_to_isaac:continue
            token=frame[slot];token[self.body_name_to_isaac[name]]=1.
            n=len(self.body_names)
            point=self.data.xpos[body]+self.data.xmat[body].reshape(3,3)@local
            token[n:n+3]=local
            token[n+3:n+6]=root_rot.T@(point-self.data.xpos[self.root])
            token[n+6:n+9]=root_rot.T@force/float(self.meta['force_limit'])
            token[-1]=1.
        def key(token):
            if token[-1]<.5:return (0.,0.,0.,0.,0.,0.,0.)
            n=len(self.body_names)
            return (1.,float(np.argmax(token[:n])),*map(float,token[n:n+9]))
        return frame[np.asarray(sorted(range(self.num_contacts),key=lambda i:key(frame[i])))].reshape(-1)

    def _surface_history(self):
        if self.contact_history_tick!=self.steps:
            self.contact_history[1:]=self.contact_history[:-1].copy()
            self.contact_history[0]=self._contact_frame()
            if self.policy_history_fresh:self.contact_history[1:]=0.
            self.contact_history_tick=self.steps
        return self.contact_history[np.asarray(self.history_offsets)].reshape(-1)

    def observation(self):
        q=self.data.qpos[self.qadr];q_asset=q[self._asset_joint_to_model()]
        q0_asset=self.q0[self._asset_joint_to_model()]
        qd=self.data.qvel[self.vadr][self.supermimic_to_model]
        root_ang=self._root_ang_vel_body();gravity=self._projected_gravity()
        if self.policy_history_fresh:
            self.actor_joint_history[:]=q_asset-q0_asset
            self.root_ang_history[:]=root_ang;self.gravity_history[:]=gravity
        else:
            self.actor_joint_history[1:]=self.actor_joint_history[:-1].copy()
            self.actor_joint_history[0]=q_asset-q0_asset
            self.root_ang_history[0]=root_ang;self.gravity_history[0]=gravity
        # The training action manager's reference is reset to measured q and
        # remains fixed for this direct-latent experiment.
        proprio=np.r_[self.root_ang_history.reshape(-1),self.gravity_history.reshape(-1),
            self.actor_joint_history.reshape(-1),self.action_history.reshape(-1),qd,
            self.reference,self.reference-q[self.supermimic_to_model],self.reference_velocity].astype(np.float32)
        forces=self._surface_history();self.policy_history_fresh=False
        result=np.r_[proprio,forces].astype(np.float32)
        if result.size!=self.meta['obs_dim']:
            raise RuntimeError(f'Expected {self.meta["obs_dim"]}-D actor input, got {result.size}')
        return result

    def _low_observation(self):
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        qd=self.data.qvel[self.vadr][self.supermimic_to_model].astype(np.float32)
        terms=(q-self.supermimic_q0,qd,self._root_ang_vel_body(),self._projected_gravity(),self.low_action)
        if self.supermimic_history_fresh:
            self.supermimic_history=[np.tile(v,(10,1)) for v in terms]
            self.supermimic_history_fresh=False
        else:
            for hist,value in zip(self.supermimic_history,terms):
                hist[:-1]=hist[1:].copy();hist[-1]=value
        return np.concatenate([x.reshape(-1) for x in self.supermimic_history]).astype(np.float32)[None]

    def _update_policy(self):
        self.last_obs=self.observation()
        with torch.inference_mode():
            actor_obs=(torch.from_numpy(self.last_obs)-self.loc)/self.scale
            raw=self.policy(actor_obs[None])[0].numpy().clip(-10.,10.)
        if not np.isfinite(raw).all():raise RuntimeError('Non-finite direct-latent surface policy output.')
        self.action_history[1:]=self.action_history[:-1].copy();self.action_history[0]=raw
        self.actions[1:]=self.actions[:-1].copy();self.actions[0]=raw
        self.last_action=raw.copy();self.last_latent=np.tanh(raw).astype(np.float32)
        goal=np.tile(np.r_[np.zeros(29),np.zeros(29),[0.,0.,-1.],np.zeros(6),self.data.qpos[2]],(1,10)).astype(np.float32)
        with torch.inference_mode():low=self.low_policy(torch.from_numpy(self._low_observation()),torch.from_numpy(self.last_latent[None]))[0].numpy()
        if not np.isfinite(low).all():raise RuntimeError('Non-finite SuperMimic decoder output.')
        self.low_action=low.astype(np.float32)
        target=self.q0.copy();target[self.supermimic_to_model]=self.supermimic_q0+self.supermimic_action_scale*self.low_action
        self.target=target


class SuperMimicOriginalSimulation(ResidualSimulation):
    """Unmodified frozen SuperMimic tracker under unobserved point forces."""
    def __init__(self,bundle,model_path,physics_dt=None):
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta.get('backend')!='supermimic-original':raise ValueError('Expected original SuperMimic bundle.')
        self.supermimic_joint_names=list(meta['supermimic_joint_names'])
        self.supermimic_to_model=np.asarray([meta['joint_names'].index(n) for n in self.supermimic_joint_names])
        self.supermimic_q0=np.asarray(meta['default_pos'],dtype=np.float32)[self.supermimic_to_model]
        self.supermimic_action_scale=np.asarray(meta['supermimic_action_scale'],dtype=np.float32)
        self.interaction_bodies=tuple(meta['interaction_bodies'])
        self.supermimic_history=[np.zeros((10,d),dtype=np.float32) for d in (29,29,3,3,29)]
        self.supermimic_history_fresh=True
        super().__init__(bundle,model_path,physics_dt)
        self._interactive_body_ids={self.model.body(n).id for n in self.interaction_bodies}

    def reset(self):
        super().reset()
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        self.reference=q.copy();self.reference_velocity=np.zeros(29,dtype=np.float32)
        self.low_action=np.zeros(29,dtype=np.float32);self.supermimic_history_fresh=True
        self.point_names=['mouse_contact'];self.points=np.asarray([self.model.body('pelvis').id],dtype=np.int32)
        self.point_offsets=np.zeros((1,3),dtype=np.float64);self.forces=np.zeros((1,3),dtype=np.float64)

    def nearest_point(self,body,hit):
        if body not in self._interactive_body_ids:return None
        self.points[0]=body
        self.point_offsets[0]=self.data.xmat[body].reshape(3,3).T@(np.asarray(hit)-self.data.xpos[body])
        return 0

    def _update_policy(self):
        q=self.data.qpos[self.qadr][self.supermimic_to_model].astype(np.float32)
        qd=self.data.qvel[self.vadr][self.supermimic_to_model].astype(np.float32)
        terms=(q-self.supermimic_q0,qd,self._body_angular_velocity(),self._projected_gravity(),self.low_action)
        if self.supermimic_history_fresh:
            self.supermimic_history=[np.tile(v,(10,1)) for v in terms]
            self.supermimic_history_fresh=False
        else:
            for hist,value in zip(self.supermimic_history,terms):hist[:-1]=hist[1:].copy();hist[-1]=value
        obs=np.concatenate([x.reshape(-1) for x in self.supermimic_history]).astype(np.float32)[None]
        goal_frame=np.r_[np.zeros(29),np.zeros(29),[0.,0.,-1.],np.zeros(6),self.data.qpos[2]].astype(np.float32)
        goal=np.tile(goal_frame,(1,10)).astype(np.float32)
        with torch.inference_mode():
            # This bundle wraps the released encoder and decoder without any
            # learned compliance residual or force observation.
            low=self.low_policy(torch.from_numpy(obs),torch.from_numpy(goal),torch.zeros((1,32)))[0].numpy()
        if not np.isfinite(low).all():raise RuntimeError('Non-finite original SuperMimic output.')
        self.low_action=low.astype(np.float32)
        self.last_action=self.low_action.copy()
        self.target=self.q0.copy();self.target[self.supermimic_to_model]=self.supermimic_q0+self.supermimic_action_scale*self.low_action

    def _body_angular_velocity(self):
        v=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,v,1)
        return v[:3].astype(np.float32)

    def _projected_gravity(self):
        return (self.data.xmat[self.root].reshape(3,3).T@np.array([0.,0.,-1.])).astype(np.float32)
