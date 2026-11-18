#!/usr/bin/env python3
"""Export the trained 3KP Jacobian latent-increment curriculum policy."""
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
TRAIN=ROOT/'outputs/20261117-supermimic-root3-jacobian-latent-increment-curriculum/161942-retry/train'
CHECKPOINT=TRAIN/'wandb/run-20261117_161854-xkzi10ut/files/checkpoint_final.pt'
SM_CHECKPOINT=TRAIN/'supermimic_frozen_checkpoint.pt'
DEPLOY_CONFIG=TRAIN/'supermimic_deploy_config.json'
BUNDLE=ROOT/'deploy/mujoco_whole_body/20261118-supermimic-3kp-jacobian-latent-increment-curriculum'
TEMPLATE=ROOT/'deploy/mujoco_whole_body/sonic/metadata.json'


class ResidualActor(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(obs_dim,512),nn.LayerNorm(512),nn.Mish(),
                                     nn.Linear(512,256),nn.LayerNorm(256),nn.Mish())
        self.mean=nn.Linear(256,32)
    def forward(self,x):return self.mean(self.features(x))


class ReleasedSuperMimic(nn.Module):
    """Pinned copy of the released SuperMimic encoder and decoder."""
    def __init__(self,checkpoint):
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
    def decode(self,obs,latent):
        x=torch.cat((latent,obs*self.obs_scale),dim=-1)
        for layer in self.decoder[:-1]:x=F.silu(layer(x))
        return self.decoder[-1](x)


class CurriculumLowLevel(nn.Module):
    """Expose the seed encoder and decoder for persistent latent integration."""
    def __init__(self,policy):
        super().__init__()
        self.encoder=nn.ModuleList([layer for layer in policy.encoder])
        self.decoder=nn.ModuleList([layer for layer in policy.decoder])
        self.register_buffer('goal_scale',policy.goal_scale.clone())
        self.register_buffer('obs_scale',policy.obs_scale.clone())
    @torch.jit.export
    def encode_goal(self,goal):
        x=goal*self.goal_scale
        for i,layer in enumerate(self.encoder):
            if i<4:x=F.silu(layer(x))
            else:return torch.tanh(layer(x))
        return x
    @torch.jit.export
    def decode_latent(self,obs,latent):
        x=torch.cat((latent,obs*self.obs_scale),dim=-1)
        for i,layer in enumerate(self.decoder):
            if i<4:x=F.silu(layer(x))
            else:return layer(x)
        return x
    def forward(self,obs,goal):return self.decode_latent(obs,self.encode_goal(goal))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from export_interactive_compliance import normalization
    if not CHECKPOINT.is_file() or not SM_CHECKPOINT.is_file() or not DEPLOY_CONFIG.is_file():
        raise FileNotFoundError('Training checkpoint or pinned SuperMimic release files are missing.')
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    task=state['cfg'].task
    if task.name!='G1SuperMimicOnlineEstimatedRoot3Crouch20ForceHistoryLatentIncrementCurriculum':
        raise ValueError(f'Unexpected task: {task.name}')
    if task.action._target_!='active_adaptation.envs.mdp.supermimic_residual.SuperMimicLatentIncrementCurriculumAction':
        raise ValueError(f'Unexpected action interface: {task.action._target_}')
    scale=float(task.action.latent_increment_scale)
    if abs(scale-.025)>1e-8:raise ValueError(f'Unexpected latent-increment scale: {scale}')
    points=task.command.force_points
    expected=[('root','pelvis'),('left_ee','left_hand_mimic'),('right_ee','right_hand_mimic')]
    if [(p.name,p.body) for p in points]!=expected:raise ValueError(f'Unexpected Jacobian point order: {points}')
    offsets=list(map(int,task.command.force_prompt_history_offsets))
    if offsets!=[0,1,2,4,8,16]:raise ValueError(f'Unexpected force-history offsets: {offsets}')

    actor_state=state['policy']['actor']
    input_dim=int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    if input_dim!=449:raise ValueError(f'Expected 395-D proprioception/latent plus 54-D force history; got {input_dim}.')
    actor=ResidualActor(input_dim).eval()
    feature_prefix='module.0.module.1.module.';mean_prefix='module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix):v for k,v in actor_state.items()
        if k.startswith(feature_prefix)},strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix):v for k,v in actor_state.items()
        if k.startswith(mean_prefix)},strict=True)

    low=ReleasedSuperMimic(SM_CHECKPOINT).eval()
    low_export=torch.jit.script(CurriculumLowLevel(low).eval())
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
        raise ValueError('Force-history normalization must be identity, matching training.')
    torch.save(norm,BUNDLE/'normalization.pt');low_export.save(str(BUNDLE/'low_level.pt'))

    meta=copy.deepcopy(template)
    meta.update(
        backend='supermimic-latent-increment-curriculum',
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(CHECKPOINT),
        supermimic_release='e11c_0220000',supermimic_checkpoint_sha256=sha(SM_CHECKPOINT),
        joint_names=model_joints,supermimic_joint_names=sm_joints,
        default_pos=[float(deploy['default_joint_pos'][i]) for i in model_to_sm],
        action_scale=[float(deploy['action_scale'][i]) for i in model_to_sm],
        supermimic_default_pos=[float(x) for x in deploy['default_joint_pos']],
        supermimic_action_scale=[float(x) for x in deploy['action_scale']],
        obs_dim=input_dim,base_obs_dim=395,force_history_offsets=offsets,
        force_history_slice=[395,input_dim],
        force_points=[{'name':p.name,'body':p.body,'offset':list(p.offset)} for p in points],
        force_limit=30.,force_slew=150.,force_budget=60.,all_body_force_points=False,
        latent_increment_scale=scale,latent_dimension=32,latent_clip=[-1.,1.],root_height=.76,
        self_collisions=False,policy_role='3KP online-Jacobian accumulated-latent curriculum teacher',
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/supermimic-online-root3-crouch20-force-history/runs/xkzi10ut',
        force_input='six root-local applied-force samples at offsets 0,1,2,4,8,16; pelvis,left hand,right hand; divided by 30 N',
        latent_interface='At reset encode a stationary 10-frame goal from the measured starting pose as seed. Each 50 Hz step accumulates 0.025*tanh(actor_output) into a persistent 32-D offset; decoder latent is clip(seed+offset,-1,1). No future GT is read after reset.',
        interactive_goal='MuJoCo starts from the nominal standing pose and encodes the current measured pose as the episode seed. This approximates the training episode-start GT latent, while no motion dataset or future target is loaded at runtime.',
        observation_terms=['root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','sonic_joint_velocity','sonic_reference','latent_reference','upper_body_forces_b'])

    torch.manual_seed(43)
    actor_probe=torch.randn(32,input_dim);obs_probe=torch.randn(8,930);goal_probe=torch.randn(8,680);latent_probe=torch.randn(8,32)
    with torch.inference_mode():
        actor_err=float((actor(actor_probe)-torch.jit.load(str(BUNDLE/'student.pt'))(actor_probe)).abs().max())
        eager_latent=low.encode(goal_probe)
        exported_latent=low_export.encode_goal(goal_probe)
        low_enc_err=float((eager_latent-exported_latent).abs().max())
        eager_dec=low.decode(obs_probe,latent_probe)
        exported_dec=low_export.decode_latent(obs_probe,latent_probe)
        low_dec_err=float((eager_dec-exported_dec).abs().max())
    if max(actor_err,low_enc_err,low_dec_err)>1e-5:
        raise RuntimeError(f'Export mismatch actor={actor_err}, encoder={low_enc_err}, decoder={low_dec_err}')
    meta.update(actor_export_max_error=actor_err,latent_encoder_export_max_error=low_enc_err,
                latent_decoder_export_max_error=low_dec_err,policy_dt=.02)
    (BUNDLE/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (BUNDLE/'README.md').write_text(f'''# SuperMimic 3KP Jacobian latent-increment curriculum policy

Exported from `{CHECKPOINT.relative_to(ROOT)}` (W&B run `xkzi10ut`). The actor has 449 inputs: 363 proprioceptive values, the current 32-D accumulated latent, and 54 values of root-local force history for pelvis and both hands. Its output is a 32-D latent increment.

## MuJoCo interaction

Run from the repository root:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-3kp-jacobian-latent-increment-curriculum
```

At reset, MuJoCo encodes ten repeats of the robot's measured starting pose and uses that as the seed latent. Every 50 Hz policy update adds `0.025 * tanh(action)` to one persistent latent offset, clips `seed + offset` to `[-1, 1]`, and decodes that latent using current SuperMimic proprioception. The offset is not restarted from the nominal pose each step. Future GT motion is not read at runtime.

The force prompt uses offsets `[0,1,2,4,8,16]`, three trained points (pelvis and both hands), root-local force components normalized by 30 N, and a 60 N total budget. In the viewer, left-drag the robot near the pelvis or either hand to apply force. The frozen low-level model is the pinned `e11c_0220000` SuperMimic release.

The training curriculum seeded each episode with the encoded initial GT motion window and gradually removed the future-GT latent base. The final stage used only the fixed episode seed plus accumulated policy increments. MuJoCo has no motion clip loaded, so its initial-pose seed is an interactive deployment approximation to that training seed. Metadata records the checkpoint hash and export parity errors.
''')
    sums={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (BUNDLE/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2)+'\n')
    print(json.dumps({'bundle':str(BUNDLE),'checkpoint':str(CHECKPOINT),'obs_dim':input_dim,
        'force_history_offsets':offsets,'latent_increment_scale':scale,'actor_export_max_error':actor_err,
        'latent_encoder_export_max_error':low_enc_err,'latent_decoder_export_max_error':low_dec_err},indent=2))


if __name__=='__main__':main()
