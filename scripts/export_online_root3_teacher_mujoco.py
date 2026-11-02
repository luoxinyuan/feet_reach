#!/usr/bin/env python3
"""Export the active 3-point GT-force teacher and its frozen low-level to MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_interactive_compliance import ResidualActor, action_metadata, normalization
from active_adaptation.utils.ceer_residual import FrozenCEER

CHECKPOINT = ROOT / (
    'outputs/20261101-ceer-online-estimated-root3-student-no-ee-16k/210200/'
    'teacher/wandb/run-20261101_210326-y3x1ilap/files/checkpoint_final.pt'
)
LOW_LEVEL_CHECKPOINT = ROOT / (
    'outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/'
    'finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt'
)
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261102-ceer-online-root3-no-ee-teacher'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/20261030-ceer-online-estimated-root3/metadata.json'


def main() -> None:
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = OmegaConf.to_container(state['cfg'], resolve=True)
    if cfg['task']['name'] != 'G1CEEROnlineEstimatedRoot3StudentNoEECompliance':
        raise ValueError(f"Unexpected task: {cfg['task']['name']}")
    if state['policy']['phase'] != 'teacher':
        raise ValueError(f"Expected GT-force teacher checkpoint, got {state['policy']['phase']}")
    points = cfg['task']['command']['force_points']
    if len(points) != 3 or [p['name'] for p in points] != ['root', 'left_ee', 'right_ee']:
        raise ValueError(f'Unexpected root3 force point order: {points}')

    actor_state = state['policy']['actor']
    input_dim = actor_state['module.0.module.1.module.0.weight'].shape[1]
    if input_dim != 363:
        raise ValueError(f'Expected 354 policy + 9 applied-force features; got {input_dim}')
    actor = ResidualActor(input_dim).eval()
    feature_prefix = 'module.0.module.1.module.'
    mean_prefix = 'module.0.module.2.module.actor_mean.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in actor_state.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in actor_state.items()
                                if k.startswith(mean_prefix)}, strict=True)

    low_state = torch.load(LOW_LEVEL_CHECKPOINT, map_location='cpu', weights_only=False)
    if low_state['policy']['last_phase'] != 'finetune':
        raise ValueError('Expected the frozen crouch-trained CEER low-level finetune checkpoint.')
    low_policy = FrozenCEER(LOW_LEVEL_CHECKPOINT, device='cpu').eval()
    low_cfg = low_state['cfg'].task.action
    template = json.loads(TEMPLATE.read_text())
    joints = template['joint_names']
    low_scales = low_cfg.action_scaling
    motor_scales = [next((float(v) for pattern, v in low_scales.items()
                          if re.fullmatch(pattern, joint)), 0.) for joint in joints]
    if len(motor_scales) != 29 or any(v <= 0 for v in motor_scales):
        raise ValueError('Could not resolve all frozen low-level motor action scales.')

    BUNDLE.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE / 'student.pt'))
    torch.save(normalization(state, input_dim), BUNDLE / 'normalization.pt')
    example_obs = torch.zeros(1, 257)
    example_reference = torch.zeros(1, 5, 29)
    torch.jit.trace(low_policy, (example_obs, example_reference), check_trace=True).save(
        str(BUNDLE / 'low_level.pt'))

    meta = copy.deepcopy(template)
    action_metadata(meta, state)
    meta.update(
        backend='ceer',
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        low_level_checkpoint_name=str(LOW_LEVEL_CHECKPOINT.relative_to(ROOT)),
        low_level_checkpoint_sha256=hashlib.sha256(LOW_LEVEL_CHECKPOINT.read_bytes()).hexdigest(),
        obs_dim=input_dim,
        force_points=points,
        force_slice=[238, 247],
        force_limit=float(cfg['task']['command'].get('force_limit', 30.)),
        force_slew=150.,
        force_budget=60.,
        all_body_force_points=True,
        observation_terms=[k for k, v in cfg['task']['observation']['policy'].items()
                           if v is None or v.get('enabled', True)],
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/ceer-online-estimated-root3-student-no-ee/runs/y3x1ilap',
        policy_role='teacher',
        force_input='MuJoCo applied force, root-local; no force estimator',
    )
    meta['action_scale'] = motor_scales
    meta['low_action_config'] = {
        'max_delay': int(low_cfg.max_delay),
        'alpha': [float(x) for x in low_cfg.alpha],
        'alpha_wide': [float(x) for x in low_cfg.alpha_wide],
        'alpha_jit_scale': float(low_cfg.alpha_jit_scale),
        'boot_protect': bool(low_cfg.boot_protect),
    }

    torch.manual_seed(211)
    actor_probe = torch.randn(32, input_dim)
    actor_export = torch.jit.load(str(BUNDLE / 'student.pt'), map_location='cpu').eval()
    actor_error = float((actor(actor_probe) - actor_export(actor_probe)).abs().max())
    low_probe = (torch.randn(16, 257), torch.randn(16, 5, 29))
    low_export = torch.jit.load(str(BUNDLE / 'low_level.pt'), map_location='cpu').eval()
    with torch.inference_mode():
        low_error = float((low_policy(*low_probe) - low_export(*low_probe)).abs().max())
    if max(actor_error, low_error) > 1e-5:
        raise RuntimeError(f'TorchScript export mismatch: actor={actor_error}, low-level={low_error}')
    meta['teacher_export_max_error'] = actor_error
    meta['low_level_export_max_error'] = low_error
    (BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'teacher_checkpoint': str(CHECKPOINT),
                      'low_level_checkpoint': str(LOW_LEVEL_CHECKPOINT),
                      'force_points': [p['name'] for p in points],
                      'teacher_export_max_error': actor_error,
                      'low_level_export_max_error': low_error}, indent=2))


if __name__ == '__main__':
    main()
