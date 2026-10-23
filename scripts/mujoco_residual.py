"""Portable causal adapters for frozen SONIC / CEER + learned residual student.

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
        if self.backend=='sonic':
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
        self.last_obs=np.zeros(381,dtype=np.float32)

    def observation(self):
        public=super().observation()
        q=self.data.qpos[self.qadr]
        return np.r_[public,self.data.qvel[self.vadr],self.reference,self.reference-q,self.reference_velocity].astype(np.float32)

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
        root_velocity=rot.T@self.data.qvel[:3]
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
        if self.backend=='sonic':
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
        if self.backend=='ceer':
            substep=self.steps%self.decimation
            idx=(self.delay-substep+self.decimation-1)//self.decimation
            delayed=self.low_actions[idx]
            self.low_applied+=self.alpha*(delayed-self.low_applied)
            self.target=self.q0+np.array(self.meta['action_scale'])*self.low_applied
            if self.boot_delay>0:self.target=self.boot_q.copy()
            self.boot_delay=max(0,self.boot_delay-1)
