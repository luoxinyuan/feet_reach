#!/usr/bin/env python3
"""Export the SuperMimic 3KP-Jacobian GT-base-latent policy for MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
TRAIN=ROOT/'outputs/20261115-supermimic-root3-crouch20-force-history-latent/095800/train'
CHECKPOINT=TRAIN/'wandb/run-20261115_095820-ebnq64fc/files/checkpoint_final.pt'
SM_CHECKPOINT=TRAIN/'supermimic_frozen_checkpoint.pt'
DEPLOY_CONFIG=TRAIN/'supermimic_deploy_config.json'
BUNDLE=ROOT/'deploy/mujoco_whole_body/20261118-supermimic-3kp-jacobian-gtbase-latent'
TEMPLATE=ROOT/'deploy/mujoco_whole_body/sonic/metadata.json'
SUPER_MIMIC_CHECKPOINT_DEFAULT=Path('/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt')


class ReleasedSuperMimic(nn.Module):
    """Strict local copy of the pinned released encoder and decoder."""
    def __init__(self,checkpoint=SUPER_MIMIC_CHECKPOINT_DEFAULT):
        super().__init__()
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)['policy']
        self.encoder=nn.ModuleList([nn.Linear(680,2048),nn.Linear(2048,1024),nn.Linear(1024,512),
            nn.Linear(512,256),nn.Linear(256,32)])
        self.decoder=nn.ModuleList([nn.Linear(962,2048),nn.Linear(2048,1024),nn.Linear(1024,512),
            nn.Linear(512,256),nn.Linear(256,29)])
        for i,layer in enumerate(self.encoder):
            layer.weight.data.copy_(p[f'encoders.motion.mlp.net.{2*i}.weight'])
            layer.bias.data.copy_(p[f'encoders.motion.mlp.net.{2*i}.bias'])
        for i,layer in enumerate(self.decoder):
            layer.weight.data.copy_(p[f'decoder.mlp.net.{2*i}.weight'])
            layer.bias.data.copy_(p[f'decoder.mlp.net.{2*i}.bias'])
        self.register_buffer('goal_scale',p['encoders.motion.scale'].clone())
        self.register_buffer('obs_scale',p['decoder.scale'].clone())
        self.requires_grad_(False);self.eval()
    def encode(self,goal):
        x=goal*self.goal_scale
        for layer in self.encoder[:-1]:x=F.silu(layer(x))
        return torch.tanh(self.encoder[-1](x))
    def decode(self,obs,z):
        x=torch.cat((z,obs*self.obs_scale),dim=-1)
        for layer in self.decoder[:-1]:x=F.silu(layer(x))
        return self.decoder[-1](x)
    def forward(self,obs,goal):return self.decode(obs,self.encode(goal))


class LatentResidualLowLevel(nn.Module):
    """Frozen SuperMimic encoder plus decoder with an added PPO latent residual."""
    def __init__(self,policy,residual_scale):
        super().__init__()
        self.encoder=nn.ModuleList([layer for layer in policy.encoder])
        self.decoder=nn.ModuleList([layer for layer in policy.decoder])
        self.register_buffer('goal_scale',policy.goal_scale.clone())
        self.register_buffer('obs_scale',policy.obs_scale.clone())
        self.residual_scale=float(residual_scale)
    @torch.jit.export
    def base_latent(self,goal):
        x=goal*self.goal_scale
        for i,layer in enumerate(self.encoder):
            if i<4:x=F.silu(layer(x))
            else:return torch.tanh(layer(x))
        return x
    def forward(self,obs,goal,residual):
        z=self.base_latent(goal)+residual
        x=torch.cat((z,obs*self.obs_scale),dim=-1)
        for i,layer in enumerate(self.decoder):
            if i<4:x=F.silu(layer(x))
            else:return layer(x)
        return x


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from export_interactive_compliance import ResidualActor, normalization
    if not CHECKPOINT.is_file() or not SM_CHECKPOINT.is_file() or not DEPLOY_CONFIG.is_file():
        raise FileNotFoundError('Training checkpoint or pinned SuperMimic release files are missing.')
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cfg=state['cfg'];task=cfg.task
    if task.name!='G1SuperMimicOnlineEstimatedRoot3Crouch20ForceHistoryLatentResidualCompliance':
        raise ValueError(f'Unexpected task: {task.name}')
    if task.action._target_!='active_adaptation.envs.mdp.supermimic_residual.SuperMimicLatentResidualAction':
        raise ValueError(f'Unexpected action interface: {task.action._target_}')
    if abs(float(task.action.latent_residual_scale)-.1)>1e-8:
        raise ValueError('Unexpected latent-residual scale.')
    points=task.command.force_points
    expected=[('root','pelvis'),('left_ee','left_hand_mimic'),('right_ee','right_hand_mimic')]
    if [(p.name,p.body) for p in points]!=expected:raise ValueError(f'Unexpected Jacobian point order: {points}')
    offsets=list(map(int,task.command.force_prompt_history_offsets))
    if offsets!=[0,1,2,4,8,16]:raise ValueError(f'Unexpected force-history offsets: {offsets}')
    actor_state=state['policy']['actor'];input_dim=int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    if input_dim!=417:raise ValueError(f'Expected 363-D proprioception plus 54-D force history, got {input_dim}.')
    actor=ResidualActor(input_dim).eval()
    actor.mean=nn.Linear(256,32)
    feature_prefix='module.0.module.1.module.';mean_prefix='module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix):v for k,v in actor_state.items()
        if k.startswith(feature_prefix)},strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix):v for k,v in actor_state.items()
        if k.startswith(mean_prefix)},strict=True)
    low=ReleasedSuperMimic(checkpoint=SM_CHECKPOINT).eval()
    low_export=torch.jit.script(LatentResidualLowLevel(low,float(task.action.latent_residual_scale)).eval())
    deploy=json.loads(DEPLOY_CONFIG.read_text());sm_joints=list(deploy['joint_names'])
    template=json.loads(TEMPLATE.read_text());model_joints=list(template['joint_names'])
    if len(sm_joints)!=29 or set(sm_joints)!=set(model_joints):raise ValueError('MuJoCo and SuperMimic joint names differ.')
    model_to_sm=[sm_joints.index(n) for n in model_joints]
    BUNDLE.mkdir(parents=True,exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE/'student.pt'))
    norm=normalization(state,input_dim)
    if tuple(norm['loc'].shape)!=(input_dim,):raise ValueError(f'Unexpected normalization: {norm["loc"].shape}')
    history_dim=len(offsets)*9
    if not torch.equal(norm['loc'][-history_dim:],torch.zeros(history_dim)) or not torch.equal(norm['scale'][-history_dim:],torch.ones(history_dim)):
        raise ValueError('Force history normalization must be identity, matching the trained input.')
    torch.save(norm,BUNDLE/'normalization.pt');low_export.save(str(BUNDLE/'low_level.pt'))
    meta=copy.deepcopy(template)
    meta.update(
        backend='supermimic-latent-force-history',checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=sha(CHECKPOINT),supermimic_release='e11c_0220000',
        supermimic_checkpoint_sha256=sha(SM_CHECKPOINT),joint_names=model_joints,
        supermimic_joint_names=sm_joints,
        default_pos=[float(deploy['default_joint_pos'][i]) for i in model_to_sm],
        action_scale=[float(deploy['action_scale'][i]) for i in model_to_sm],
        supermimic_default_pos=[float(x) for x in deploy['default_joint_pos']],
        supermimic_action_scale=[float(x) for x in deploy['action_scale']],
        obs_dim=input_dim,base_obs_dim=input_dim-history_dim,force_history_offsets=offsets,
        force_history_slice=[input_dim-history_dim,input_dim],
        force_points=[{'name':p.name,'body':p.body,'offset':list(p.offset)} for p in points],
        force_limit=30.,force_slew=150.,force_budget=60.,all_body_force_points=True,
        latent_residual_scale=float(task.action.latent_residual_scale),root_height=.76,
        self_collisions=False,policy_role='3KP online-Jacobian latent-residual teacher',
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/supermimic-online-root3-crouch20-force-history-latent/runs/ebnq64fc',
        force_input='six root-local applied-force samples at offsets 0,1,2,4,8,16; pelvis,left hand,right hand; divided by 30 N',
        latent_interface='encode 10-frame goal with frozen SuperMimic encoder; add 0.1*tanh(actor_output); decode using frozen proprioceptive decoder',
        interactive_goal='MuJoCo interactive mode supplies a stationary nominal-pose 10-frame goal. The original benchmark used future GT motion frames; this default interactive mode does not load a motion dataset.',
        observation_terms=['root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','sonic_joint_velocity','sonic_reference','upper_body_forces_b'])
    torch.manual_seed(43)
    actor_probe=torch.randn(32,input_dim);obs_probe=torch.randn(8,930);goal_probe=torch.randn(8,680);res_probe=torch.randn(8,32)
    with torch.inference_mode():
        actor_err=float((actor(actor_probe)-torch.jit.load(str(BUNDLE/'student.pt'))(actor_probe)).abs().max())
        eager=low(obs_probe,goal_probe)
        native=low.decode(obs_probe,low.encode(goal_probe)+float(task.action.latent_residual_scale)*res_probe.tanh())
        exported=low_export(obs_probe,goal_probe,float(task.action.latent_residual_scale)*res_probe.tanh())
        low_err=float((native-exported).abs().max());base_err=float((eager-low_export(obs_probe,goal_probe,torch.zeros_like(res_probe))).abs().max())
    if max(actor_err,low_err,base_err)>1e-5:raise RuntimeError(f'Export mismatch actor={actor_err}, decoder={low_err}, base={base_err}')
    meta.update(actor_export_max_error=actor_err,latent_decoder_export_max_error=low_err,
                zero_residual_low_level_export_max_error=base_err,policy_dt=.02)
    (BUNDLE/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (BUNDLE/'README.md').write_text(f'''# SuperMimic 3KP Jacobian GT-base-latent residual policy

Exported from `{CHECKPOINT.relative_to(ROOT)}` (W&B run `ebnq64fc`). The actor consumes 363 proprioceptive values and six root-local force-history samples for pelvis and both hands (417 inputs total). The frozen SuperMimic encoder maps a 10-frame goal to one 32-D base latent. At each 50 Hz update the policy adds `0.1 * tanh(actor_output)` to that latent and passes it to the frozen decoder.

## MuJoCo interaction

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-3kp-jacobian-gtbase-latent
```

Left-drag the robot to apply forces at the nearest of the trained three Jacobian points: pelvis, left hand, or right hand. The tool uses the trained 30 N per-point cap, 60 N total budget, 150 N/s slew, and force-history offsets `[0,1,2,4,8,16]`.

## Reference conditioning

The Isaac benchmark for this checkpoint supplied the next ten ground-truth motion frames to the SuperMimic encoder on every step. This is reference-conditioned tracking, not a force-only motion-generation result; its reported benchmark MAE must not be interpreted as force-only performance. The interactive MuJoCo viewer has no motion dataset, so it uses a stationary nominal-pose goal to produce the base latent. It preserves the learned residual and force-history path, but does not reproduce the benchmark's future-GT input. No motion dataset or Isaac package is loaded at runtime.
''')
    sums={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (BUNDLE/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2)+'\n')
    print(json.dumps({'bundle':str(BUNDLE),'checkpoint':str(CHECKPOINT),'obs_dim':input_dim,
        'force_history_offsets':offsets,'actor_export_max_error':actor_err,
        'latent_decoder_export_max_error':low_err,'zero_residual_low_level_export_max_error':base_err},indent=2))


if __name__=='__main__':main()
