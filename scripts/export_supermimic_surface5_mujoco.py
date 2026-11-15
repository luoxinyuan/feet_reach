#!/usr/bin/env python3
"""Export the observable-surface 5-contact SuperMimic latent policy for MuJoCo."""
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
CHECKPOINT=ROOT/'outputs/20261114_230110-surface-observable5-latent-teacher/wandb/run-20261114_230217-dt54sa0h/files/checkpoint_final.pt'
SM_ROOT=ROOT/'outputs/20261114_230110-surface-observable5-latent-teacher'
SM_RELEASE=Path('/home/xl521/supermimic_force_transition')
SM_CHECKPOINT=SM_ROOT/'ckpt_e11c_0220000.pt'
SM_DEPLOY=SM_ROOT/'deploy_config.json'
BUNDLE=ROOT/'deploy/mujoco_whole_body/20261115-supermimic-surface5-latent'
TEMPLATE=ROOT/'deploy/mujoco_whole_body/sonic/metadata.json'

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


class SurfaceLatentLowLevel(nn.Module):
    def __init__(self,policy):
        super().__init__()
        self.encoder=nn.ModuleList([layer for layer in policy.encoder])
        self.decoder=nn.ModuleList([layer for layer in policy.decoder])
        self.register_buffer('goal_scale',policy.goal_scale.clone())
        self.register_buffer('obs_scale',policy.obs_scale.clone())
    @torch.jit.export
    def base_latent(self,goal):
        x=goal*self.goal_scale
        for i,layer in enumerate(self.encoder):
            if i<4:x=F.silu(layer(x))
            else:return torch.tanh(layer(x))
        return x
    def forward(self,obs,goal,residual):
        z=(self.base_latent(goal)+residual).clamp(-1.,1.)
        x=torch.cat((z,obs*self.obs_scale),dim=-1)
        for i,layer in enumerate(self.decoder):
            if i<4:x=F.silu(layer(x))
            else:return layer(x)
        return x


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from export_interactive_compliance import normalization
    os.environ['SUPER_MIMIC_ROOT']=str(SM_RELEASE)
    sm_module=ROOT/'active_adaptation/utils/supermimic.py'
    spec=importlib.util.spec_from_file_location('feet_reach_supermimic',sm_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ReleasedSuperMimic=module.ReleasedSuperMimic
    if not CHECKPOINT.is_file() or not SM_CHECKPOINT.is_file() or not SM_DEPLOY.is_file():
        raise FileNotFoundError('The selected policy or frozen SuperMimic release files are missing.')
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cfg=state['cfg'];task=cfg.task
    if task.name!='G1SuperMimicSurfaceObservable5':raise ValueError(f'Unexpected task: {task.name}')
    if bool(task.command.student_train):raise ValueError('Expected the teacher actor checkpoint, not student adaptation.')
    if int(task.command.max_contacts)!=5 or float(task.command.spring_k)!=80.:
        raise ValueError('Checkpoint command does not match the selected 5-contact spring method.')
    if float(task.action.latent_scale)!=2. or int(task.action.interpolation_frames)!=10:
        raise ValueError('Checkpoint action is not the 10-step latent endpoint policy.')
    deploy=json.loads(SM_DEPLOY.read_text())
    sm_joints=list(deploy['joint_names'])
    template=json.loads(TEMPLATE.read_text());model_joints=list(template['joint_names'])
    if len(ASSET_BODIES)!=33 or len(ASSET_JOINTS)!=29:raise AssertionError('Bad G1 asset ordering table.')
    if len(sm_joints)!=29 or set(sm_joints)!=set(model_joints) or set(ASSET_JOINTS)!=set(model_joints):
        raise ValueError('MuJoCo, Isaac asset and released SuperMimic joint names must match.')
    actor_state=state['policy']['actor']
    input_dim=int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    base_dim=372;force_dim=5*(len(ASSET_BODIES)+10)*6
    if (base_dim+force_dim,input_dim)!=(1662,1662):
        raise ValueError(f'Expected actor observation 372+1290=1662, got {base_dim}+{force_dim}={input_dim}')
    actor=ResidualActor(input_dim).eval()
    feature_prefix='module.0.module.1.module.';mean_prefix='module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix):v for k,v in actor_state.items()
        if k.startswith(feature_prefix)},strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix):v for k,v in actor_state.items()
        if k.startswith(mean_prefix)},strict=True)
    BUNDLE.mkdir(parents=True,exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE/'student.pt'))
    norm=normalization(state,input_dim)
    if tuple(norm['loc'].shape)!=(input_dim,):raise ValueError('Unexpected actor normalization shape.')
    torch.save(norm,BUNDLE/'normalization.pt')
    low=ReleasedSuperMimic(checkpoint=SM_CHECKPOINT).eval()
    low_export=torch.jit.script(SurfaceLatentLowLevel(low).eval())
    low_export.save(str(BUNDLE/'low_level.pt'))

    model_to_sm=[sm_joints.index(n) for n in model_joints]
    model_to_asset=[ASSET_JOINTS.index(n) for n in model_joints]
    meta=copy.deepcopy(template)
    meta['default_pos']=[float(deploy['default_joint_pos'][i]) for i in model_to_sm]
    meta['action_scale']=[float(deploy['action_scale'][i]) for i in model_to_sm]
    placeholders=[('contact_0','pelvis'),('contact_1','torso_link'),
                  ('contact_2','left_hand_mimic'),('contact_3','right_hand_mimic'),
                  ('contact_4','head_mimic')]
    meta.update(
        backend='supermimic-surface-latent',
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(CHECKPOINT),
        supermimic_release='e11c_0220000',supermimic_checkpoint_sha256=sha(SM_CHECKPOINT),
        joint_names=model_joints,asset_joint_names=ASSET_JOINTS,asset_body_names=ASSET_BODIES,
        supermimic_joint_names=sm_joints,
        supermimic_default_pos=[float(x) for x in deploy['default_joint_pos']],
        supermimic_action_scale=[float(x) for x in deploy['action_scale']],
        obs_dim=input_dim,base_obs_dim=base_dim,force_history_offsets=[0,1,2,4,8,16],
        max_contacts=5,force_points=[{'name':n,'body':b,'offset':[0.,0.,0.]} for n,b in placeholders],
        force_limit=30.,force_slew=150.,force_budget=60.,spring_k=80.,spring_damping=4.,
        root_height=.76,latent_scale=2.,interpolation_frames=10,
        force_input='5 contact tokens x (33 body one-hot + local point xyz + root-relative point xyz + root-frame force xyz / 30 N + active bit); six newest-first frames at offsets 0,1,2,4,8,16',
        contact_interface='Mouse drag selects one upper-body surface point and pulls its body-attached point with an 80 N/m spring and 4 N s/m damping. Training actor accepts up to five simultaneous contacts; interactive viewer currently drives one selected contact at a time.',
        force_cap='30 N per point, 60 N sum of point-force magnitudes, 150 N/s slew per point',
        low_level_input='Released SuperMimic 10-frame term-major proprioception; reference-free stationary 10-frame goal plus a 32-D latent residual endpoint',
        policy_role='teacher',
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/supermimic-surface-compliance/runs/dt54sa0h',
        observation_terms=['root_ang_vel_history','projected_gravity_history','joint_pos_history',
            'prev_actions','sonic_joint_velocity','latent_reference','surface_contacts'],
    )
    torch.manual_seed(32)
    actor_probe=torch.randn(16,input_dim)
    obs_probe=torch.randn(4,930);goal_probe=torch.randn(4,680);latent_probe=torch.randn(4,32)
    with torch.inference_mode():
        actor_error=float((actor(actor_probe)-torch.jit.load(str(BUNDLE/'student.pt'))(actor_probe)).abs().max())
        eager=SurfaceLatentLowLevel(low).eval()(obs_probe,goal_probe,latent_probe)
        scripted=low_export(obs_probe,goal_probe,latent_probe)
        low_error=float((eager-scripted).abs().max())
        base_error=float((low.encode(goal_probe)-low_export.base_latent(goal_probe)).abs().max())
    if max(actor_error,low_error,base_error)>1e-5:
        raise RuntimeError(f'Export mismatch actor={actor_error}, low={low_error}, base latent={base_error}')
    meta['actor_export_max_error']=actor_error
    meta['supermimic_export_max_error']=low_error
    meta['base_latent_export_max_error']=base_error
    (BUNDLE/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    readme=f'''# SuperMimic observable-surface 5-contact latent policy\n\nThis bundle exports the completed teacher policy from `{CHECKPOINT.relative_to(ROOT)}` (W&B run `dt54sa0h`). It uses a reference-free SuperMimic low-level tracker and a 32-D latent endpoint residual. The actor output is interpolated toward its endpoint over 10 control updates; each update decodes one latent into the 29-joint low-level target. No future motion clip or joint ground-truth is loaded at runtime.\n\n## MuJoCo interaction\n\nRun from the repository root:\n\n```bash\npython scripts/mujoco_whole_body_student.py --policy supermimic-surface5-latent\n```\n\nLeft-drag an upper-body surface to apply a point spring (80 N/m, 4 N s/m damping). The selected point moves with its body. Force is capped at 30 N per point and 60 N total, with a 150 N/s slew rate. The viewer currently drives one selected contact at a time; the policy input preserves the trained five-contact token layout and can represent up to five active contacts.\n\n## Input alignment\n\nThe high-level actor receives 372 proprioception/latent values and six histories of five 43-value tokens (1662 total). Each contact token is Isaac asset body one-hot, body-local application point, root-relative point, root-frame applied force divided by 30 N, and a visibility bit. Inactive slots are zero and contacts are canonically sorted, matching training.\n\nThe frozen SuperMimic release is `e11c_0220000`. The low-level goal is generated from the reset reference and current root state; it is not a prerecorded future trajectory.\n'''
    (BUNDLE/'README.md').write_text(readme)
    files={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (BUNDLE/'SHA256SUMS.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps({'bundle':str(BUNDLE),'checkpoint':str(CHECKPOINT),'obs_dim':input_dim,
        'asset_body_count':len(ASSET_BODIES),'latent_dim':32,'actor_export_max_error':actor_error,
        'supermimic_export_max_error':low_error,'base_latent_export_max_error':base_error},indent=2))


if __name__=='__main__':main()
