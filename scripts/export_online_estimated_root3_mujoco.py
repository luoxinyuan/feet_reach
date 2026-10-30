#!/usr/bin/env python3
"""Export the latest causal 3-point force-estimator student to standalone MuJoCo."""
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
CHECKPOINT = ROOT / 'outputs/ceer-online-estimated-root3/full_20261029_131216/finetune/wandb/run-20261030_003020-4e69gvyj/files/checkpoint_final.pt'
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261030-ceer-online-estimated-root3-v2'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/20261028-ceer-ee-root3/metadata.json'
LOW_LEVEL = ROOT / 'deploy/mujoco_whole_body/20261026-segmented-multi10/low_level.pt'
FROZEN_CEER = ROOT / '.cache/ceer/checkpoint_final.pt'


class Root3ForceEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(930, 512), nn.LayerNorm(512), nn.Mish(),
                          nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish()),
            nn.Linear(256, 9))

    def forward(self, x):
        force = self.net(x).reshape(x.shape[0], 3, 3)
        return (force / force.norm(p=2, dim=-1, keepdim=True).clamp_min(1.)).flatten(1)


def main():
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    low_state = torch.load(FROZEN_CEER, map_location='cpu', weights_only=False)
    cfg = OmegaConf.to_container(state['cfg'], resolve=True)
    if cfg['task']['name'] != 'G1CEEROnlineEstimatedRoot3Compliance':
        raise ValueError(f"Unexpected training task: {cfg['task']['name']}")
    points = cfg['task']['command']['force_points']
    if len(points) != 3 or [p['name'] for p in points] != ['root', 'left_ee', 'right_ee']:
        raise ValueError(f'Unexpected root3 force point order: {points}')

    actor_state = state['policy']['actor']
    actor = ResidualActor(actor_state['module.0.module.1.module.0.weight'].shape[1]).eval()
    feature_prefix = 'module.0.module.1.module.'
    mean_prefix = 'module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in actor_state.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in actor_state.items()
                                if k.startswith(mean_prefix)}, strict=True)
    if actor.features[0].in_features != 363:
        raise ValueError(f'Expected 354 policy observations + 9 estimated force values; got {actor.features[0].in_features}')
    if low_state['policy']['last_phase'] != 'finetune':
        raise ValueError('Expected the frozen CEER finetune checkpoint used during residual training.')
    low_action = low_state['cfg'].task.action
    joints = json.loads(TEMPLATE.read_text())['joint_names']
    low_action_scale = [next((float(v) for pattern, v in low_action.action_scaling.items()
                              if re.fullmatch(pattern, joint)), 0.) for joint in joints]
    if len(low_action_scale) != 29 or any(scale <= 0 for scale in low_action_scale):
        raise ValueError('Could not resolve all 29 frozen CEER action scales.')

    estimator = Root3ForceEstimator().eval()
    prefix = 'module.'
    estimator.load_state_dict({k.removeprefix(prefix): v for k, v in state['policy']['estimator'].items()
                               if k.startswith(prefix)}, strict=True)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE / 'student.pt'))
    torch.jit.script(estimator).save(str(BUNDLE / 'estimator.pt'))
    torch.save(normalization(state, 363), BUNDLE / 'normalization.pt')
    (BUNDLE / 'low_level.pt').write_bytes(LOW_LEVEL.read_bytes())

    meta = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    action_metadata(meta, state)
    meta.update(
        backend='dense-estimator', checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        obs_dim=363, force_points=points, force_slice=[238, 247],
        force_limit=30., force_slew=150., all_body_force_points=True,
        estimator={'history_frames': 10, 'history_frame_dim': 93, 'input_dim': 930,
                   'force_dim': 9, 'force_scale_N': 30., 'force_frame': 'root-local',
                   'points': [p['name'] for p in points]},
        observation_terms=list(cfg['task']['observation']['policy'].keys()),
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/ceer-online-estimated-root3-compliance/runs/4e69gvyj',
    )
    # The residual checkpoint's action config controls its reference generation;
    # the frozen CEER action manager controls how its low-level output reaches motors.
    # MuJoCo needs the latter here to match CEERResidualAction's nested manager.
    meta['residual_reference_action_scale'] = [float(v) for v in state['cfg'].task.action.action_scaling.values()]
    meta['action_scale'] = low_action_scale
    meta['low_action_config'] = {
        'max_delay': int(low_action.max_delay),
        'alpha': [float(x) for x in low_action.alpha],
        'alpha_wide': [float(x) for x in low_action.alpha_wide],
        'alpha_jit_scale': float(low_action.alpha_jit_scale),
        'boot_protect': bool(low_action.boot_protect),
    }
    meta['low_level_checkpoint_sha256'] = hashlib.sha256(FROZEN_CEER.read_bytes()).hexdigest()
    meta['native_student_max_error'] = 0.
    meta['force_estimator_export_max_error'] = 0.
    torch.manual_seed(41)
    actor_probe = torch.randn(32, 363)
    actor_error = float((actor(actor_probe) - torch.jit.load(str(BUNDLE / 'student.pt'))(actor_probe)).abs().max())
    estimator_probe = torch.randn(32, 930)
    estimator_error = float((estimator(estimator_probe) - torch.jit.load(str(BUNDLE / 'estimator.pt'))(estimator_probe)).abs().max())
    if max(actor_error, estimator_error) > 1e-6:
        raise RuntimeError(f'TorchScript mismatch: actor={actor_error}, estimator={estimator_error}')
    meta['native_student_max_error'] = actor_error
    meta['force_estimator_export_max_error'] = estimator_error
    (BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'actor_error': actor_error,
                      'estimator_error': estimator_error, 'force_points': [p['name'] for p in points],
                      'low_action_scale': low_action_scale, 'low_action_config': meta['low_action_config']}, indent=2))


if __name__ == '__main__':
    main()
