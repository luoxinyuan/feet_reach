"""Fit a fixed surface keypoint atlas from per-contact rollout traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.audit / 'seed0/result.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    if meta['rows'] != manifest['clips']:
        raise ValueError('Audit rows do not match the input manifest')
    groups = defaultdict(list)
    for i, row in enumerate(meta['rows']):
        groups[row['category']].append(i)
    rng = np.random.default_rng(20261122)
    fit_ids = sorted(int(i) for group in groups.values()
                     for i in rng.choice(group, (len(group)+1)//2, replace=False))
    eval_ids = sorted(set(range(len(meta['rows']))) - set(fit_ids))

    def source_id(row):
        return row['dataset'], row['source_motion_id']

    fit_sources = {source_id(meta['rows'][i]) for i in fit_ids}
    if any(source_id(meta['rows'][i]) in fit_sources for i in eval_ids):
        raise ValueError('Fit and holdout must not share source clips')
    contacts = defaultdict(list)
    for folder in sorted(path for path in args.audit.glob('seed*') if path.is_dir()):
        meta_seed = json.loads((folder/'result.json').read_text())
        assert meta_seed['rows'] == meta['rows']
        names = meta_seed['contact_trace_body_names']
        with np.load(folder/'traces.npz') as d:
            pos = d['point_local_pos_m'][:, fit_ids]
            body = d['point_body_ids'][:, fit_ids].astype(int)
            visible = (np.linalg.norm(d['point_forces_world_N'][:,fit_ids],axis=-1)>.05
                       ) & d['valid'][:,fit_ids,None]
            new = visible.copy()
            changed = (body[1:] != body[:-1]) | np.any(pos[1:] != pos[:-1], axis=-1)
            new[1:] &= changed | ~visible[:-1]
            for name_id in np.unique(body[new]):
                contacts[names[name_id]].extend(pos[new & (body == name_id)].tolist())

    ranked = sorted(contacts, key=lambda name: (-len(contacts[name]), name))
    if len(ranked) < 9:
        raise ValueError('Fewer than nine body links were observed')
    points = []
    for name in ranked[:9]:
        local = np.array(contacts[name], dtype=np.float64)
        # Restrict the representative to a point that was actually sampled on the mesh.
        distances = np.linalg.norm(local[:,None] - local[None,:], axis=-1)
        medoid = local[distances.mean(axis=1).argmin()]
        points.append(dict(body=name, offset=medoid.tolist(), observed_episodes=len(local)))
    (args.output/'layout.json').write_text(json.dumps(dict(
        points=points,fit_rows=fit_ids,holdout_rows=eval_ids,
        fit_source=str(args.audit.resolve()),
        episode_counts={name:len(contacts[name]) for name in ranked}),indent=2)+'\n')
    manifest['clips'] = [meta['rows'][i] for i in eval_ids]
    (args.output/'holdout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
