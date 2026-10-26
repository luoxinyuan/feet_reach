"""29-DOF residual references drive the original frozen CEER finetune actor."""
import torch
import hydra
from tensordict import TensorDict
from .action import ActionManager
from active_adaptation.utils.sonic import ResidualReference, JOINT_NAMES
from active_adaptation.utils.ceer_residual import FrozenCEER, TARGET_JOINT_NAMES
from active_adaptation.utils.usd_kinematics import USDKinematics
from active_adaptation.utils.symmetry import joint_space_symmetry

class CEERHistory:
    def __init__(self,n,device):
        self.q=torch.zeros(n,5,29,device=device)
        self.physics_q=torch.zeros(n,2,29,device=device)
        self.values=[self.q,self.physics_q]
        self.fresh=torch.ones(n,dtype=torch.bool,device=device)

    def reset(self,ids,q):
        self.q[ids]=q[ids,None]
        self.physics_q[ids]=q[ids,None]
        self.fresh[ids]=True

    def push(self,q):
        self.q[:,1:]=self.q[:,:-1].clone()
        self.q[:,0]=self.physics_q.mean(1)
        self.q[self.fresh]=q[self.fresh,None]
        self.fresh.zero_()
        return self.q.flatten(1)

class CEERResidualAction(ActionManager):
    def __init__(self,env,reference_speed=2.,max_reference_error=.6,
                 ceer_checkpoint=None,ceer_force_limit=5.,**kwargs):
        super().__init__(env)
        if abs(env.step_dt-.02)>1e-6 or env.cfg.robot.name != 'g1_col_full':
            raise ValueError('CEER requires its own G1 motor parameters and 50Hz control.')
        self.joint_names=JOINT_NAMES
        self.joint_ids=[self.asset.joint_names.index(n) for n in JOINT_NAMES]
        self.action_dim=29
        self.low_policy=FrozenCEER(ceer_checkpoint,device=self.device)
        self.low_action_manager=hydra.utils.instantiate(self.low_policy.action_cfg,env=env)
        assert self.low_action_manager.joint_names == self.joint_names
        self.default_joint_pos=self.asset.data.default_joint_pos.clone()
        self.nominal=self.default_joint_pos[0,self.joint_ids].clone()
        self.offset=torch.zeros_like(self.default_joint_pos)
        self.action_scaling=self.low_action_manager.action_scaling
        limits=self.asset.data.soft_joint_pos_limits[:,self.joint_ids]
        self.reference=ResidualReference(self.num_envs,limits[...,0],limits[...,1],env.step_dt,reference_speed,max_reference_error)
        self.target_order=[JOINT_NAMES.index(n) for n in TARGET_JOINT_NAMES]
        self.kinematics=USDKinematics(self.asset.cfg.spawn.usd_path,JOINT_NAMES,self.device)
        self.history=CEERHistory(self.num_envs,self.device)
        self.action_buf=torch.zeros(self.num_envs,3,29,device=self.device)
        self.applied_action=torch.zeros(self.num_envs,29,device=self.device)
        self.low_action=torch.zeros_like(self.applied_action)
        self.age=torch.zeros(self.num_envs,1,device=self.device)
        self.ceer_force_limit=float(ceer_force_limit)
        self.last_policy_obs=torch.zeros(self.num_envs,257,device=self.device)
        self.last_joint_reference=torch.zeros(self.num_envs,5,29,device=self.device)
        env._post_step_callbacks.append(self.post_step)

    def symmetry_transforms(self):
        return joint_space_symmetry(self.asset,self.joint_names)

    @torch.no_grad()
    def reset(self,env_ids):
        q=self.asset.data.joint_pos[:,self.joint_ids]
        self.reference.reset(env_ids,q)
        self.history.reset(env_ids,q)
        self.low_action_manager.reset(env_ids)
        self.action_buf[env_ids]=0; self.applied_action[env_ids]=0; self.low_action[env_ids]=0
        self.age[env_ids]=0
        self.last_policy_obs[env_ids]=0; self.last_joint_reference[env_ids]=0

    def post_step(self,substep):
        self.history.physics_q[:,substep%2]=self.asset.data.joint_pos[:,self.joint_ids]

    def low_inputs(self,joint_history):
        data=self.asset.data
        # Existing CEER root interface: measured height/velocity and current
        # heading (1,0), with no external desired root trajectory.
        command=torch.cat(((data.root_pos_w[:,2:3]-self.env.scene.env_origins[:,2:3]),
            data.root_lin_vel_b[:,:2],torch.ones_like(self.age),torch.zeros_like(self.age),
            torch.full_like(self.age,self.ceer_force_limit)),-1)
        wrists=self.kinematics.interface(self.reference.q)
        boot=(1-self.age/25).clamp(0,1)
        obs=torch.cat((boot,command,wrists,data.root_ang_vel_b,data.projected_gravity_b,
            joint_history,self.low_action_manager.action_buf[:,:3].flatten(1)),-1)
        target=self.reference.q[:,self.target_order][:,None].expand(-1,5,-1)
        assert obs.shape==(self.num_envs,257)
        return obs,target

    @torch.no_grad()
    def __call__(self,tensordict,substep):
        if substep==0:
            raw=tensordict['action']
            self.action_buf=self.action_buf.roll(1,1);self.action_buf[:,0]=raw
            self.applied_action.copy_(raw.tanh())
            q=self.asset.data.joint_pos[:,self.joint_ids]
            self.reference.update(raw,q)
            history=self.history.push(q)
            obs,target=self.low_inputs(history)
            self.last_policy_obs.copy_(obs);self.last_joint_reference.copy_(target)
            self.low_action.copy_(self.low_policy(obs,target))
            self.age.add_(1)
        # Preserve original CEER action scaling/filtering/delay and boot PD.
        self.low_action_manager(TensorDict({'action':self.low_action},[self.num_envs]),substep)
