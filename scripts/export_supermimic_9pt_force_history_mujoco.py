#!/usr/bin/env python3
"""Export the completed SuperMimic + 9-point force-history policy for MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / (
    'outputs/20261110-supermimic-online-estimated-9pt-crouch20-force-history-gpu567/'
    'train/wandb/run-20261110_121638-28wvjxpb/files/checkpoint_final.pt'
)
SUPER_MIMIC_ROOT = Path('/home/xl521/supermimic_force_transition')
SUPER_MIMIC_CHECKPOINT = SUPER_MIMIC_ROOT / 'pretrained/e11c_0220000/ckpt_e11c_0220000.pt'
DEPLOY_CONFIG = SUPER_MIMIC_ROOT / 'pretrained/e11c_0220000/deploy_config.json'
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261110-supermimic-9pt-crouch20-force-history'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/sonic/metadata.json'
POINT_BODIES = ['torso_link'] + [f'{side}_{part}' for side in ('left', 'right')
    for part in ('shoulder_roll_link', 'elbow_link', 'wrist_roll_link', 'hand_mimic')]


def main() -> None:
    from export_interactive_compliance import ResidualActor, normalization
    from active_adaptation.utils.supermimic import ReleasedSuperMimic

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = OmegaConf.to_container(state['cfg'], resolve=True)
    task = cfg['task']
    expected_task = 'G1SuperMimicOnlineEstimated9PointNoEECrouch20ForceHistoryCompliance'
    if task['name'] != expected_task:
        raise ValueError(f"Unexpected training task: {task['name']}")
    if state['policy'].get('phase') != 'teacher':
        raise ValueError(f"Expected completed teacher-stage policy; got {state['policy'].get('phase')}")
    command = task['command']
    if int(command['expected_force_points']) != len(POINT_BODIES):
        raise ValueError(f"Expected nine Jacobian force points; config has {command['expected_force_points']}")
    offsets = list(map(int, command['force_prompt_history_offsets']))
    if offsets != [0, 1, 2, 4, 8, 16]:
        raise ValueError(f'Unexpected force-history offsets: {offsets}')
    points = [{'name': name, 'body': name, 'offset': [0., 0., 0.]}
              for name in POINT_BODIES]

    deploy = json.loads(DEPLOY_CONFIG.read_text())
    supermimic_joints = list(deploy['joint_names'])
    if len(supermimic_joints) != 29 or len(set(supermimic_joints)) != 29:
        raise ValueError('Released SuperMimic deploy config must contain 29 unique joints.')
    template_meta = json.loads(TEMPLATE.read_text())
    if set(supermimic_joints) != set(template_meta['joint_names']):
        raise ValueError('SuperMimic joints do not match the bundled G1 MuJoCo model.')

    actor_state = state['policy']['actor']
    input_dim = int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    history_dim = len(offsets) * len(POINT_BODIES) * 3
    base_dim = input_dim - history_dim
    if (base_dim, input_dim) != (354, 516):
        raise ValueError(f'Expected 354-D proprioception + 162-D force history; got {base_dim}+{history_dim}')
    actor = ResidualActor(input_dim).eval()
    feature_prefix = 'module.0.module.1.module.'
    mean_prefix = 'module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in actor_state.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in actor_state.items()
                                if k.startswith(mean_prefix)}, strict=True)

    BUNDLE.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE / 'student.pt'))
    norm = normalization(state, input_dim)
    if tuple(norm['loc'].shape) != (input_dim,):
        raise ValueError(f'Unexpected normalization shape: {norm["loc"].shape}')
    if not torch.equal(norm['loc'][base_dim:], torch.zeros(history_dim)) or not torch.equal(
            norm['scale'][base_dim:], torch.ones(history_dim)):
        raise ValueError('Force-history features must retain identity normalization.')
    torch.save(norm, BUNDLE / 'normalization.pt')

    low_policy = ReleasedSuperMimic(checkpoint=SUPER_MIMIC_CHECKPOINT).eval()
    low_probe = (torch.randn(8, 930), torch.randn(8, 680))
    with torch.inference_mode():
        torch.jit.trace(low_policy, low_probe, check_trace=True).save(str(BUNDLE / 'low_level.pt'))
    low_export = torch.jit.load(str(BUNDLE / 'low_level.pt'), map_location='cpu').eval()

    model_joints = list(template_meta['joint_names'])
    model_to_supermimic = [supermimic_joints.index(name) for name in model_joints]
    template_meta['default_pos'] = [float(deploy['default_joint_pos'][i]) for i in model_to_supermimic]
    template_meta['action_scale'] = [float(deploy['action_scale'][i]) for i in model_to_supermimic]
    template_meta.update(
        backend='supermimic-force-history',
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        supermimic_release='e11c_0220000',
        supermimic_checkpoint_sha256=hashlib.sha256(SUPER_MIMIC_CHECKPOINT.read_bytes()).hexdigest(),
        joint_names=model_joints,
        supermimic_joint_names=supermimic_joints,
        supermimic_action_scale=[float(x) for x in deploy['action_scale']],
        obs_dim=input_dim,
        base_obs_dim=base_dim,
        force_slice=[base_dim, base_dim + 3*len(POINT_BODIES)],
        force_history_offsets=offsets,
        force_history_slice=[base_dim, input_dim],
        force_points=points,
        force_limit=30.,
        force_slew=150.,
        force_budget=60.,
        all_body_force_points=True,
        root_height=0.76,
        reference_speed=float(task['action']['reference_speed']),
        max_reference_error=float(task['action']['max_reference_error']),
        observation_terms=[k for k, v in task['observation']['policy'].items()
                           if v is None or v.get('enabled', True)],
        policy_role='teacher',
        wandb_run=f"https://wandb.ai/luoxinyuan-duke-university/supermimic-online-estimated-9pt-crouch20-force-history/runs/{state['wandb']['id']}",
        force_input='six root-local samples at offsets 0,1,2,4,8,16; torso and four left/right arm links; divided by 30 N',
        interactive_goal='10-frame stationary upright goal and persistent learned 29-joint reference residual; no motion clip is loaded',
        supermimic_proprioception='10 oldest-first frames, term-major: q-default, qd, root angular velocity, projected gravity, previous low-level action',
    )

    torch.manual_seed(41)
    actor_probe = torch.randn(32, input_dim)
    actor_export = torch.jit.load(str(BUNDLE / 'student.pt'), map_location='cpu').eval()
    with torch.inference_mode():
        actor_error = float((actor(actor_probe) - actor_export(actor_probe)).abs().max())
        low_error = float((low_policy(*low_probe) - low_export(*low_probe)).abs().max())
    if max(actor_error, low_error) > 1e-5:
        raise RuntimeError(f'TorchScript export mismatch: actor={actor_error}, low-level={low_error}')
    template_meta['actor_export_max_error'] = actor_error
    template_meta['supermimic_export_max_error'] = low_error
    (BUNDLE / 'metadata.json').write_text(json.dumps(template_meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'checkpoint': str(CHECKPOINT),
                      'supermimic_checkpoint': str(SUPER_MIMIC_CHECKPOINT),
                      'force_points': POINT_BODIES, 'obs_dim': input_dim,
                      'base_obs_dim': base_dim, 'force_history_offsets': offsets,
                      'actor_export_max_error': actor_error,
                      'supermimic_export_max_error': low_error}, indent=2))


if __name__ == '__main__':
    main()
