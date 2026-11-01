#!/usr/bin/env python3
"""Export the completed 9-point online-estimated CEER residual to MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from export_interactive_compliance import ResidualActor, action_metadata, normalization

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'outputs/20261030-ceer-online-estimated-9pt-no-ee/20261030_170115/finetune/wandb/run-20261031_035529-qaxgsgg1/files/checkpoint_final.pt'
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261101-ceer-online-estimated-9pt'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/20261030-ceer-online-estimated-root3/metadata.json'
LOW_LEVEL = ROOT / 'deploy/mujoco_whole_body/20261030-ceer-online-estimated-root3/low_level.pt'
FROZEN_CEER = ROOT / '.cache/ceer/checkpoint_final.pt'
POINT_BODIES = ['torso_link'] + [f'{side}_{part}' for side in ('left', 'right')
    for part in ('shoulder_roll_link', 'elbow_link', 'wrist_roll_link', 'hand_mimic')]


class NinePointForceEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(930, 512), nn.LayerNorm(512), nn.Mish(),
                          nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish()),
            nn.Linear(256, 27))

    def forward(self, x):
        forces = self.net(x).reshape(x.shape[0], 9, 3)
        return (forces / forces.norm(p=2, dim=-1, keepdim=True).clamp_min(1.)).flatten(1)


def main():
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state['cfg'] = OmegaConf.to_container(state['cfg'], resolve=True)
    if state['cfg']['task']['name'] != 'G1CEEROnlineEstimated9PointNoEECompliance':
        raise ValueError(f"Unexpected task: {state['cfg']['task']['name']}")
    if state['policy']['phase'] != 'finetune':
        raise ValueError(f"Expected finetune checkpoint, got {state['policy']['phase']}")

    low_state = torch.load(FROZEN_CEER, map_location='cpu', weights_only=False)
    template = json.loads(TEMPLATE.read_text())
    actor_state = state['policy']['actor']
    actor = ResidualActor(actor_state['module.0.module.1.module.0.weight'].shape[1]).eval()
    feature_prefix = 'module.0.module.1.module.'
    mean_prefix = 'module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in actor_state.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in actor_state.items()
                                if k.startswith(mean_prefix)}, strict=True)
    if actor.features[0].in_features != 381:
        raise ValueError(f'Expected 354 policy features + 27 force estimates, got {actor.features[0].in_features}')

    estimator = NinePointForceEstimator().eval()
    prefix = 'module.'
    estimator.load_state_dict({k.removeprefix(prefix): v for k, v in state['policy']['estimator'].items()
                               if k.startswith(prefix)}, strict=True)

    joints = template['joint_names']
    low_action = low_state['cfg'].task.action
    low_scales = low_action.action_scaling
    action_scale = [next((float(v) for pattern, v in low_scales.items()
                          if re.fullmatch(pattern, joint)), 0.) for joint in joints]
    if len(action_scale) != 29 or any(scale <= 0 for scale in action_scale):
        raise ValueError('Could not resolve the 29 frozen CEER motor action scales.')

    BUNDLE.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE / 'student.pt'))
    torch.jit.script(estimator).save(str(BUNDLE / 'estimator.pt'))
    torch.save(normalization(state, 381), BUNDLE / 'normalization.pt')
    (BUNDLE / 'low_level.pt').write_bytes(LOW_LEVEL.read_bytes())

    points = [{'name': name, 'body': name, 'offset': [0., 0., 0.]}
              for name in POINT_BODIES]
    meta = copy.deepcopy(template)
    action_metadata(meta, state)
    meta.update(
        backend='dense-estimator', checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        obs_dim=381, force_points=points, force_slice=[354, 381], force_limit=30.,
        force_slew=150., all_body_force_points=False,
        estimator={'history_frames': 10, 'history_frame_dim': 93, 'input_dim': 930,
                   'force_dim': 27, 'force_scale_N': 30., 'force_frame': 'root-local',
                   'points': POINT_BODIES},
        observation_terms=[k for k, v in state['cfg']['task']['observation']['policy'].items()
                           if v is None or v.get('enabled', True)],
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/ceer-online-estimated-9pt-compliance/runs/qaxgsgg1',
    )
    meta['residual_reference_action_scale'] = [float(v) for v in state['cfg']['task']['action']['action_scaling'].values()]
    meta['action_scale'] = action_scale
    meta['low_action_config'] = {
        'max_delay': int(low_action.max_delay),
        'alpha': [float(x) for x in low_action.alpha],
        'alpha_wide': [float(x) for x in low_action.alpha_wide],
        'alpha_jit_scale': float(low_action.alpha_jit_scale),
        'boot_protect': bool(low_action.boot_protect),
    }
    meta['low_level_checkpoint_sha256'] = hashlib.sha256(FROZEN_CEER.read_bytes()).hexdigest()

    torch.manual_seed(43)
    actor_x, estimator_x = torch.randn(32, 381), torch.randn(32, 930)
    actor_error = float((actor(actor_x) - torch.jit.load(str(BUNDLE / 'student.pt'))(actor_x)).abs().max())
    estimator_error = float((estimator(estimator_x) - torch.jit.load(str(BUNDLE / 'estimator.pt'))(estimator_x)).abs().max())
    if max(actor_error, estimator_error) > 1e-6:
        raise RuntimeError(f'TorchScript export mismatch: actor={actor_error}, estimator={estimator_error}')
    meta['native_student_max_error'] = actor_error
    meta['force_estimator_export_max_error'] = estimator_error
    (BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'checkpoint': str(CHECKPOINT),
        'force_points': POINT_BODIES, 'actor_error': actor_error,
        'estimator_error': estimator_error, 'low_action_scale': action_scale}, indent=2))


if __name__ == '__main__':
    main()
