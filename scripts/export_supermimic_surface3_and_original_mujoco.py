#!/usr/bin/env python3
"""Export the 3-contact direct-latent student and unmodified SuperMimic baseline."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
CHECKPOINT=ROOT/'outputs/20261117-supermimic-surface3-direct-latent-compliancemix-gpu234/train/wandb/run-20261116_235926-syfbqljg/files/checkpoint_final.pt'
SM_ROOT=ROOT/'outputs/20261114_230110-surface-observable5-latent-teacher'
SM_CHECKPOINT=SM_ROOT/'ckpt_e11c_0220000.pt'
SM_DEPLOY=ROOT/'outputs/20261117-supermimic-surface3-direct-latent-compliancemix-gpu234/train/supermimic_deploy_config.json'
SURFACE_BUNDLE=ROOT/'deploy/mujoco_whole_body/20261118-supermimic-surface3-direct-latent'
BASE_BUNDLE=ROOT/'deploy/mujoco_whole_body/20261118-supermimic-original-arbitrary-contact'
TEMPLATE=ROOT/'deploy/mujoco_whole_body/sonic/metadata.json'
BASE_LOW=ROOT/'deploy/mujoco_whole_body/20261115-supermimic-surface5-latent/low_level.pt'

ASSET_BODIES=[
    'pelvis','left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link',
    'left_ankle_pitch_link','left_ankle_roll_link','right_hip_pitch_link','right_hip_roll_link',
    'right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link',
    'waist_yaw_link','waist_roll_link','torso_link','left_shoulder_pitch_link',
    'left_shoulder_roll_link','left_shoulder_yaw_link','left_elbow_link','left_wrist_roll_link',
    'left_wrist_pitch_link','left_wrist_yaw_link','right_shoulder_pitch_link',
    'right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link','right_wrist_roll_link',
    'right_wrist_pitch_link','right_wrist_yaw_link','left_hand_mimic','right_hand_mimic','head_mimic',
]
ASSET_JOINTS=[
    'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint',
    'left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint',
    'right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
    'waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint',
    'left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint',
    'left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint',
    'right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint',
    'right_wrist_pitch_joint','right_wrist_yaw_joint',
]


class ResidualActor(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(obs_dim,512),nn.LayerNorm(512),nn.Mish(),
                                     nn.Linear(512,256),nn.LayerNorm(256),nn.Mish())
        self.mean=nn.Linear(256,32)
    def forward(self,x):return self.mean(self.features(x))


class DirectLatentDecoder(nn.Module):
    def __init__(self,policy):
        super().__init__()
        self.decoder=nn.ModuleList([layer for layer in policy.decoder])
        self.register_buffer('obs_scale',policy.obs_scale.clone())
    def forward(self,obs,latent):
        x=torch.cat((latent,obs*self.obs_scale),dim=-1)
        for i,layer in enumerate(self.decoder):
            if i<4:x=F.silu(layer(x))
            else:return layer(x)
        return x


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from export_interactive_compliance import normalization
    os.environ['SUPER_MIMIC_ROOT']='/home/xl521/supermimic_force_transition'
    if not CHECKPOINT.is_file() or not SM_CHECKPOINT.is_file() or not SM_DEPLOY.is_file():
        raise FileNotFoundError('Surface policy checkpoint or frozen SuperMimic release is missing.')
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cfg=state['cfg'];task=cfg.task
    if task.name!='G1SuperMimicSurfaceObservable3DirectLatentComplianceMix':
        raise ValueError(f'Unexpected task: {task.name}')
    if int(task.command.max_contacts)!=3 or float(task.command.spring_k)!=80.:
        raise ValueError('Checkpoint is not the full-body 3-contact, 80 N/m spring policy.')
    if task.action._target_!='active_adaptation.envs.mdp.supermimic_residual.SuperMimicDirectLatentAction':
        raise ValueError(f'Unexpected action interface: {task.action._target_}')
    actor_state=state['policy']['actor']
    obs_dim=int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    if obs_dim!=1137:raise ValueError(f'Expected the 1137-D 3-contact actor, got {obs_dim}.')
    actor=ResidualActor(obs_dim).eval()
    feature_prefix='module.0.module.1.module.';mean_prefix='module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix):v for k,v in actor_state.items()
        if k.startswith(feature_prefix)},strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix):v for k,v in actor_state.items()
        if k.startswith(mean_prefix)},strict=True)

    sm_module=ROOT/'active_adaptation/utils/supermimic.py'
    spec=importlib.util.spec_from_file_location('feet_reach_supermimic_export',sm_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    low=module.ReleasedSuperMimic(checkpoint=SM_CHECKPOINT).eval()
    direct_low=torch.jit.script(DirectLatentDecoder(low).eval())

    deploy=json.loads(SM_DEPLOY.read_text())
    sm_joints=list(deploy['joint_names'])
    template=json.loads(TEMPLATE.read_text());model_joints=list(template['joint_names'])
    if len(ASSET_BODIES)!=33 or len(ASSET_JOINTS)!=29:raise AssertionError('Bad G1 ordering tables.')
    if len(sm_joints)!=29 or set(sm_joints)!=set(model_joints) or set(ASSET_JOINTS)!=set(model_joints):
        raise ValueError('MuJoCo, Isaac and SuperMimic joint names do not match.')
    model_to_sm=[sm_joints.index(n) for n in model_joints]

    SURFACE_BUNDLE.mkdir(parents=True,exist_ok=True)
    torch.jit.script(actor).save(str(SURFACE_BUNDLE/'student.pt'))
    norm=normalization(state,obs_dim)
    if tuple(norm['loc'].shape)!=(obs_dim,):raise ValueError('Unexpected normalization shape.')
    torch.save(norm,SURFACE_BUNDLE/'normalization.pt')
    direct_low.save(str(SURFACE_BUNDLE/'low_level.pt'))
    meta=copy.deepcopy(template)
    meta.update(
        backend='supermimic-surface-direct-latent',checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=sha(CHECKPOINT),supermimic_release='e11c_0220000',
        supermimic_checkpoint_sha256=sha(SM_CHECKPOINT),joint_names=model_joints,
        asset_joint_names=ASSET_JOINTS,asset_body_names=ASSET_BODIES,supermimic_joint_names=sm_joints,
        supermimic_default_pos=[float(x) for x in deploy['default_joint_pos']],
        default_pos=[float(deploy['default_joint_pos'][i]) for i in model_to_sm],
        action_scale=[float(deploy['action_scale'][i]) for i in model_to_sm],
        supermimic_action_scale=[float(x) for x in deploy['action_scale']],obs_dim=obs_dim,
        base_obs_dim=363,force_history_offsets=[0,1,2,4,8,16],max_contacts=3,
        force_points=[{'name':f'contact_{i}','body':b,'offset':[0.,0.,0.]} for i,b in enumerate(('pelvis','torso_link','left_hand_mimic'))],
        interaction_bodies=ASSET_BODIES,
        force_limit=30.,force_slew=150.,force_budget=60.,spring_k=80.,spring_damping=4.,
        root_height=.76,self_collisions=False,latent_interface='direct: tanh(actor_output) is passed as the complete 32-D decoder latent; no base-latent addition or interpolation',
        contact_interface='Mouse drag selects any of the 33 represented G1 bodies and applies a body-attached point spring (80 N/m, 4 N s/m). Viewer drives one contact at a time; policy token layout preserves three simultaneous slots.',
        force_input='3 canonical contact tokens x (33 body one-hot + body-local point xyz + root-relative point xyz + root-frame force xyz / 30 N + active bit), six newest-first frames at offsets 0,1,2,4,8,16',
        policy_role='direct-latent compliance teacher',
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/supermimic-surface-compliancemix/runs/syfbqljg',
        observation_terms=['root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','sonic_joint_velocity','sonic_reference','surface_contacts'])
    torch.manual_seed(32)
    actor_probe=torch.randn(16,obs_dim);obs_probe=torch.randn(4,930);latent_probe=torch.randn(4,32)
    with torch.inference_mode():
        actor_err=float((actor(actor_probe)-torch.jit.load(str(SURFACE_BUNDLE/'student.pt'))(actor_probe)).abs().max())
        decoder_err=float((DirectLatentDecoder(low)(obs_probe,latent_probe)-direct_low(obs_probe,latent_probe)).abs().max())
        native_err=float((low.decode(obs_probe,latent_probe)-direct_low(obs_probe,latent_probe)).abs().max())
    if max(actor_err,decoder_err,native_err)>1e-5:raise RuntimeError((actor_err,decoder_err,native_err))
    meta.update(actor_export_max_error=actor_err,decoder_export_max_error=decoder_err,
                native_decoder_max_error=native_err,policy_dt=.02)
    (SURFACE_BUNDLE/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (SURFACE_BUNDLE/'README.md').write_text(f'''# SuperMimic full-body surface 3-contact direct-latent policy

Exported from `{CHECKPOINT.relative_to(ROOT)}` (W&B run `syfbqljg`), trained on ComplianceMix-29h-v1. The actor directly predicts the complete 32-D SuperMimic decoder latent as `tanh(actor_output)`. It does not add a motion-reference base latent and does not interpolate actions.

## MuJoCo interaction

Run from the repository root:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-surface3-direct-latent
```

Left-drag any visible represented body surface. The selected point follows its body and receives an 80 N/m spring with 4 N s/m damping; the cap is 30 N at the point and 60 N total, with 150 N/s slew. The desktop interface supports one active mouse drag at a time. The actor retains the trained three-contact input format.

The force prompt has 6 frame samples at offsets `[0,1,2,4,8,16]`. Each slot contains body one-hot, body-local point, root-relative point, root-frame force normalized by 30 N, and visibility. The frozen low-level release is `e11c_0220000`; it receives current 10-frame proprioception and the direct latent only. No reference-motion dataset is needed at runtime.

The actor input is 363 proprioceptive values plus 774 contact-history values (1137 total). Metadata records the training checkpoint hash, action interface, joint orders, and export parity checks.
''')

    BASE_BUNDLE.mkdir(parents=True,exist_ok=True)
    (BASE_BUNDLE/'low_level.pt').write_bytes(BASE_LOW.read_bytes())
    base=copy.deepcopy(meta)
    base.update(backend='supermimic-original',checkpoint_name='Released SuperMimic e11c_0220000 (frozen low-level only)',
        checkpoint_sha256=sha(SM_CHECKPOINT),obs_dim=0,base_obs_dim=0,force_history_offsets=[],max_contacts=1,
        force_points=[{'name':'mouse_contact','body':'pelvis','offset':[0.,0.,0.]}],
        interaction_bodies=ASSET_BODIES,force_budget=60.,policy_role='original frozen SuperMimic tracker; no compliance student or force input',
        force_input='none: external MuJoCo point force affects physics but is not provided to the policy',
        contact_interface='Left-drag any of the 33 represented G1 body surfaces. One body-attached physical point spring is applied; this baseline has no force prompt or learned compliance actor.',
        latent_interface='Original frozen SuperMimic motion encoder produces the latent from a stationary nominal-pose goal; the original decoder maps proprioception and that encoded latent to joint targets.')
    (BASE_BUNDLE/'metadata.json').write_text(json.dumps(base,indent=2)+'\n')
    (BASE_BUNDLE/'README.md').write_text('''# Original SuperMimic in MuJoCo with arbitrary point interaction

This bundle runs the released frozen SuperMimic motion encoder and decoder directly, with no compliance student and no force tokens. It provides a stationary nominal-pose goal and lets MuJoCo physical forces perturb any represented robot body.

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-original-arbitrary-contact
```

Left-drag any visible represented G1 body surface. The point follows the body and receives an 80 N/m spring with 4 N s/m damping, capped at 30 N and slewed at 150 N/s. The force changes the simulated robot dynamics but is not an input to the original SuperMimic policy. This makes it a physical-interaction baseline against the learned surface-compliance model.

The base controller is the frozen `e11c_0220000` SuperMimic release. Its 10-frame proprioceptive history and stationary nominal-pose goal are sent through the original encoder/decoder path. There are no motion clips or training data loaded at runtime.
''')

    for bundle in (SURFACE_BUNDLE,BASE_BUNDLE):
        sums={p.name:sha(p) for p in bundle.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
        (bundle/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2)+'\n')
    print(json.dumps({'surface_bundle':str(SURFACE_BUNDLE),'base_bundle':str(BASE_BUNDLE),
        'actor_export_max_error':actor_err,'decoder_export_max_error':decoder_err,
        'native_decoder_max_error':native_err,'actor_obs_dim':obs_dim},indent=2))


if __name__=='__main__':main()
