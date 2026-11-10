#!/usr/bin/env python3
"""Export the completed SONIC 9-point Jacobian force-history policy for MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    'outputs/20261109-sonic-online-estimated-9pt-crouch20-force-history/'
    '234gpu-234159/train/wandb/run-20261109_234301-gcvu2d6a/files/checkpoint_final.pt'
)
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261110-sonic-9pt-crouch20-force-history'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/sonic/metadata.json'
POINT_BODIES = ['torso_link'] + [f'{side}_{part}' for side in ('left', 'right')
    for part in ('shoulder_roll_link', 'elbow_link', 'wrist_roll_link', 'hand_mimic')]


def main() -> None:
    from export_interactive_compliance import ResidualActor, action_metadata, normalization

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = OmegaConf.to_container(state['cfg'], resolve=True)
    task = cfg['task']
    if task['name'] != 'G1SonicOnlineEstimated9PointNoEECrouch20ForceHistoryCompliance':
        raise ValueError(f"Unexpected task: {task['name']}")
    if state['policy'].get('phase') != 'teacher':
        raise ValueError(f"Expected completed teacher policy; got {state['policy'].get('phase')}")
    command = task['command']
    if int(command['expected_force_points']) != len(POINT_BODIES):
        raise ValueError(f"Expected nine configured force points: {command['expected_force_points']}")
    offsets = list(map(int, command['force_prompt_history_offsets']))
    if offsets != [0, 1, 2, 4, 8, 16]:
        raise ValueError(f'Unexpected force-history offsets: {offsets}')
    points = [{'name': name, 'body': name, 'offset': [0., 0., 0.]}
              for name in POINT_BODIES]

    actor_state = state['policy']['actor']
    input_dim = actor_state['module.0.module.1.module.0.weight'].shape[1]
    history_dim = len(offsets) * len(POINT_BODIES) * 3
    base_dim = input_dim - history_dim
    if (base_dim, input_dim) != (354, 516):
        raise ValueError(f'Expected 354 proprioception + 162 history features; got {base_dim}+{history_dim}')
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
        raise ValueError('Force-history features must use identity normalization.')
    torch.save(norm, BUNDLE / 'normalization.pt')

    meta = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    action_metadata(meta, state)
    meta.update(
        backend='sonic-force-history',
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        obs_dim=input_dim,
        base_obs_dim=base_dim,
        force_history_offsets=offsets,
        force_history_slice=[base_dim, input_dim],
        force_points=points,
        force_limit=30.,
        force_slew=150.,
        force_budget=60.,
        all_body_force_points=True,
        reference_speed=float(task['action']['reference_speed']),
        max_reference_error=float(task['action']['max_reference_error']),
        observation_terms=[k for k, v in task['observation']['policy'].items()
                           if v is None or v.get('enabled', True)],
        wandb_run_id=state.get('wandb', {}).get('id'),
        policy_role='teacher',
        force_input='six root-local samples at offsets 0,1,2,4,8,16; torso and four left/right arm links; divided by 30 N',
    )
    (BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    exported = torch.jit.load(str(BUNDLE / 'student.pt'), map_location='cpu').eval()
    torch.manual_seed(41)
    probe = torch.randn(32, input_dim)
    with torch.inference_mode():
        export_error = float((actor(probe) - exported(probe)).abs().max())
    if export_error > 1e-6:
        raise RuntimeError(f'TorchScript export differs from source actor: {export_error}')
    print(json.dumps({'bundle': str(BUNDLE), 'checkpoint': str(CHECKPOINT),
                      'force_points': POINT_BODIES, 'obs_dim': input_dim,
                      'base_obs_dim': base_dim, 'force_history_offsets': offsets,
                      'export_max_error': export_error}, indent=2))


if __name__ == '__main__':
    main()
