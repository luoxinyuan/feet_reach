#!/usr/bin/env python3
"""Export trained 3-point and 23-point compliance students for standalone MuJoCo."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ROOT3_CHECKPOINT = ROOT / 'outputs/ceer-segmented-jacobian-residual/20261027_173438_ee-root3/wandb/run-20261027_173540-a88dyzbq/files/checkpoint_final.pt'
DENSE_CHECKPOINT = ROOT / 'outputs/ceer-dense-force/full_20261027_230935/finetune/wandb/run-20261028_043733-4wm17n5y/files/checkpoint_final.pt'
ROOT3_BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261028-ceer-ee-root3'
DENSE_BUNDLE = ROOT / 'deploy/mujoco_whole_body/20261028-ceer-dense23'


class ResidualActor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(obs_dim, 512), nn.LayerNorm(512), nn.Mish(),
                                      nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish())
        self.mean = nn.Linear(256, 29)

    def forward(self, x):
        return self.mean(self.features(x))


class ForceEstimator(nn.Module):
    def __init__(self, dim=69):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(930, 512), nn.LayerNorm(512), nn.Mish(),
                          nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish()),
            nn.Linear(256, dim))

    def forward(self, x):
        force = self.net(x).reshape(x.shape[0], 23, 3)
        return (force / force.norm(p=2, dim=-1, keepdim=True).clamp_min(1.)).flatten(1)


def checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    state['cfg'] = OmegaConf.to_container(state['cfg'], resolve=True)
    return state


def actor_from_checkpoint(state):
    source = state['policy']['actor']
    obs_dim = source['module.0.module.1.module.0.weight'].shape[1]
    actor = ResidualActor(obs_dim).eval()
    feature_prefix = 'module.0.module.1.module.'
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in source.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    mean_prefix = 'module.0.module.2.module.actor_mean.'
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in source.items()
                                if k.startswith(mean_prefix)}, strict=True)
    return actor


def normalization(state, dim):
    env = state['env']
    if 'transforms.2.loc' in env:
        loc, scale = env['transforms.2.loc'], env['transforms.2.scale']
    else:
        stats = state['vecnorm']['_extra_state']
        count = stats['policy_count']
        loc = stats['policy_sum'] / count
        eps = 1e-2  # TorchRL VecNorm default.
        scale = (stats['policy_ssq'] / count - loc.square()).clamp_min(eps).sqrt().clamp_min(eps)
    if loc.numel() < dim and scale.numel() == loc.numel():
        # Dense-force actor receives the estimator output as a separate, unnormalized key.
        padding = dim - loc.numel()
        loc = torch.cat((loc, torch.zeros(padding)))
        scale = torch.cat((scale, torch.ones(padding)))
    if tuple(loc.shape) != (dim,) or tuple(scale.shape) != (dim,):
        raise ValueError(f'Unexpected policy normalization shapes: {loc.shape}, {scale.shape}, expected {dim}')
    return {'loc': loc.cpu(), 'scale': scale.cpu()}


def action_metadata(meta, state):
    cfg = state['cfg']['task']['action']
    joints = meta['joint_names']
    scales = cfg['action_scaling']
    meta['action_scale'] = [next((float(v) for pat, v in scales.items() if re.fullmatch(pat, joint)), 0.)
                            for joint in joints]
    alpha = list(cfg['alpha'])
    meta['low_action_config'] = {'max_delay': cfg['max_delay'], 'alpha': alpha,
                                 'alpha_wide': alpha, 'boot_protect': bool(cfg.get('boot_protect', False))}
    meta['reference_speed'] = float(cfg['reference_speed'])
    meta['max_reference_error'] = float(cfg['max_reference_error'])
    meta['ceer_force_limit'] = float(cfg['ceer_force_limit'])


def base_metadata(state, bundle_name, backend, force_points, obs_dim, force_slice):
    template = json.loads((ROOT / 'deploy/mujoco_whole_body/20261026-segmented-multi10/metadata.json').read_text())
    meta = copy.deepcopy(template)
    path = ROOT3_CHECKPOINT if bundle_name == ROOT3_BUNDLE.name else DENSE_CHECKPOINT
    meta.update(backend=backend, checkpoint_name=f'{path.parent.parent.name}/{path.name}',
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), obs_dim=obs_dim,
                force_points=force_points, force_slice=force_slice, force_limit=30., force_slew=150.)
    meta['observation_terms'] = [k for k, v in state['cfg']['task']['observation']['policy'].items()
                                 if v.get('enabled', True)]
    action_metadata(meta, state)
    return meta


def export_actor(state, bundle, meta):
    bundle.mkdir(parents=True, exist_ok=True)
    actor = actor_from_checkpoint(state)
    torch.jit.script(actor).save(str(bundle / 'student.pt'))
    norm = normalization(state, actor.features[0].in_features)
    torch.save(norm, bundle / 'normalization.pt')
    torch.manual_seed(17)
    x = torch.randn(32, actor.features[0].in_features)
    loaded = torch.jit.load(str(bundle / 'student.pt'))
    error = float((actor(x) - loaded(x)).abs().max())
    if error > 1e-6:
        raise RuntimeError(f'Actor export mismatch: {error}')
    meta['native_student_max_error'] = error
    (bundle / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root3-checkpoint', type=Path, default=ROOT3_CHECKPOINT)
    parser.add_argument('--dense-checkpoint', type=Path, default=DENSE_CHECKPOINT)
    parser.add_argument('--dense-manifest', type=Path, default=ROOT / 'artifacts/ceer_dense_link/full23/val/manifest.json')
    parser.add_argument('--only', choices=['all', 'root3', 'dense23'], default='all')
    args = parser.parse_args()
    low_bundle = ROOT / 'deploy/mujoco_whole_body/20261026-segmented-multi10'

    if args.only in ('all', 'root3'):
        state = checkpoint(args.root3_checkpoint)
        points = state['cfg']['task']['command']['force_points']
        if len(points) != 3:
            raise ValueError(f'Expected exactly 3 force points, got {len(points)}')
        meta = base_metadata(state, ROOT3_BUNDLE.name, 'segmented', points, 363, [238, 247])
        actor = export_actor(state, ROOT3_BUNDLE, meta)
        for name in ('low_level.pt',):
            target = ROOT3_BUNDLE / name
            if not target.exists():
                target.write_bytes((low_bundle / name).read_bytes())
        (ROOT3_BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(f'Exported root3 actor input={actor.features[0].in_features} bundle={ROOT3_BUNDLE}')

    if args.only in ('all', 'dense23'):
        state = checkpoint(args.dense_checkpoint)
        points = json.loads(args.dense_manifest.read_text())['body_names']
        if len(points) != 23:
            raise ValueError(f'Expected exactly 23 force points, got {len(points)}')
        force_points = [{'name': name, 'body': name, 'offset': [0., 0., 0.]} for name in points]
        meta = base_metadata(state, DENSE_BUNDLE.name, 'dense-estimator', force_points, 423, [354, 423])
        meta['all_body_force_points'] = True
        meta['estimator'] = {'history_frames': 10, 'history_frame_dim': 93, 'input_dim': 930,
                             'force_dim': 69, 'force_scale_N': 30., 'force_frame': 'root-local'}
        actor = export_actor(state, DENSE_BUNDLE, meta)
        estimator = ForceEstimator().eval()
        source = state['policy']['estimator']
        prefix = 'module.net.'
        estimator.net.load_state_dict({k.removeprefix(prefix): v for k, v in source.items()
                                       if k.startswith(prefix)}, strict=True)
        torch.jit.script(estimator).save(str(DENSE_BUNDLE / 'estimator.pt'))
        torch.manual_seed(29)
        x = torch.randn(32, 930)
        exported = torch.jit.load(str(DENSE_BUNDLE / 'estimator.pt'))
        est_error = float((estimator(x) - exported(x)).abs().max())
        actor_input = torch.randn(32, 423)
        actor_error = float((actor(actor_input) - torch.jit.load(str(DENSE_BUNDLE / 'student.pt'))(actor_input)).abs().max())
        if max(est_error, actor_error) > 1e-6:
            raise RuntimeError(f'Dense export mismatch: actor={actor_error}, estimator={est_error}')
        meta['force_estimator_export_max_error'] = est_error
        meta['native_student_max_error'] = actor_error
        for name in ('low_level.pt',):
            target = DENSE_BUNDLE / name
            if not target.exists():
                target.write_bytes((low_bundle / name).read_bytes())
        (DENSE_BUNDLE / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(f'Exported dense23 actor input={actor.features[0].in_features} estimator input=930 bundle={DENSE_BUNDLE}')

    for bundle in (ROOT3_BUNDLE, DENSE_BUNDLE):
        if args.only != 'all' and bundle not in ((ROOT3_BUNDLE,) if args.only == 'root3' else (DENSE_BUNDLE,)):
            continue
        sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bundle.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
        (bundle / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')


if __name__ == '__main__':
    main()
