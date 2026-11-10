#!/usr/bin/env python3
"""Export the SuperMimic + 3-point force-history policy for standalone MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / (
    'outputs/20261109-supermimic-root3-crouch20-force-history/174413/train/'
    'wandb/run-20261109_174510-x4elg15b/files/checkpoint_final.pt'
)
SUPER_MIMIC_ROOT = Path('/home/xl521/supermimic_force_transition')
SUPER_MIMIC_CHECKPOINT = SUPER_MIMIC_ROOT / 'pretrained/e11c_0220000/ckpt_e11c_0220000.pt'
DEPLOY_CONFIG = SUPER_MIMIC_ROOT / 'pretrained/e11c_0220000/deploy_config.json'
BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261110-supermimic-root3-crouch20-force-history'
TEMPLATE = ROOT / 'deploy/mujoco_whole_body/sonic/metadata.json'
TEMPLATE_META = json.loads(TEMPLATE.read_text())


def main() -> None:
    from export_interactive_compliance import ResidualActor, normalization
    from active_adaptation.utils.supermimic import ReleasedSuperMimic

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = state['cfg']
    task = cfg.task
    if task.name != 'G1SuperMimicOnlineEstimatedRoot3Crouch20ForceHistoryCompliance':
        raise ValueError(f'Unexpected training task: {task.name}')
    if state['policy'].get('phase') != 'teacher':
        raise ValueError(f"Expected teacher-stage actor, got {state['policy'].get('phase')}")

    points = task.command.force_points
    expected_points = [('root', 'pelvis'), ('left_ee', 'left_hand_mimic'),
                       ('right_ee', 'right_hand_mimic')]
    if [(p.name, p.body) for p in points] != expected_points:
        raise ValueError(f'Unexpected 3KP order or bodies: {points}')
    offsets = list(map(int, task.command.force_prompt_history_offsets))
    if offsets != [0, 1, 2, 4, 8, 16]:
        raise ValueError(f'Unexpected force history offsets: {offsets}')

    deploy = json.loads(DEPLOY_CONFIG.read_text())
    supermimic_joints = list(deploy['joint_names'])
    if len(supermimic_joints) != 29 or len(set(supermimic_joints)) != 29:
        raise ValueError('Released SuperMimic deploy config must contain 29 unique joints.')
    if set(supermimic_joints) != set(TEMPLATE_META['joint_names']):
        raise ValueError('SuperMimic joint names do not match the bundled G1 MuJoCo model.')

    actor_state = state['policy']['actor']
    input_dim = int(actor_state['module.0.module.1.module.0.weight'].shape[1])
    history_dim = len(offsets) * 9
    base_dim = input_dim - history_dim
    if (base_dim, input_dim) != (354, 408):
        raise ValueError(f'Expected 354-D policy + 54-D force history; got {base_dim}+{history_dim}')
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
        raise ValueError('Force history must have identity normalization, matching training.')
    torch.save(norm, BUNDLE / 'normalization.pt')

    low_policy = ReleasedSuperMimic(checkpoint=SUPER_MIMIC_CHECKPOINT).eval()
    low_probe = (torch.randn(8, 930), torch.randn(8, 680))
    with torch.inference_mode():
        torch.jit.trace(low_policy, low_probe, check_trace=True).save(str(BUNDLE / 'low_level.pt'))
    low_export = torch.jit.load(str(BUNDLE / 'low_level.pt'), map_location='cpu').eval()

    meta = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    model_joints = list(meta['joint_names'])
    model_to_supermimic = [supermimic_joints.index(name) for name in model_joints]
    meta['default_pos'] = [float(deploy['default_joint_pos'][i]) for i in model_to_supermimic]
    meta['action_scale'] = [float(deploy['action_scale'][i]) for i in model_to_supermimic]
    meta.update(
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
        force_history_offsets=offsets,
        force_history_slice=[base_dim, input_dim],
        force_points=[{'name': p.name, 'body': p.body, 'offset': list(p.offset)} for p in points],
        force_limit=30.,
        force_slew=150.,
        force_budget=60.,
        all_body_force_points=True,
        root_height=0.76,
        reference_speed=float(task.action.reference_speed),
        max_reference_error=float(task.action.max_reference_error),
        observation_terms=[k for k, v in task.observation.policy.items()
                           if v is None or v.get('enabled', True)],
        policy_role='teacher',
        wandb_run='https://wandb.ai/luoxinyuan-duke-university/supermimic-online-root3-crouch20-force-history/runs/x4elg15b',
        force_input='six root-local applied-force samples at offsets 0,1,2,4,8,16; pelvis,left_ee,right_ee; divided by 30 N',
        interactive_goal='10-frame stationary upright standing goal plus the learned 29-joint residual reference; no motion dataset is loaded',
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
    meta['actor_export_max_error'] = actor_error
    meta['supermimic_export_max_error'] = low_error
    (BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BUNDLE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    (BUNDLE / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'checkpoint': str(CHECKPOINT),
                      'supermimic_checkpoint': str(SUPER_MIMIC_CHECKPOINT),
                      'joint_names': supermimic_joints, 'force_history_offsets': offsets,
                      'obs_dim': input_dim, 'actor_export_max_error': actor_error,
                      'supermimic_export_max_error': low_error}, indent=2))


if __name__ == '__main__':
    main()
