"""Build exact offline force-prompt caches for the fixed compliance manifest."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.compliance_eval.protocol import load_arrays
from scripts.offline_force_baseline.solver import solve_sequence
from active_adaptation.utils.segmented_guidance import geometry_jacobians


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def solve_record(record, points, geometry, k, budget, output):
    source = (ROOT / 'dataset' / record['dataset']).resolve()
    meta, arrays = load_arrays(source)
    clip = int(record['clip'])
    lo = int(meta['starts'][clip])
    n = int(record['frames']) - 1
    if n < 1 or int(meta['ends'][clip]) < lo + n + 1:
        raise ValueError(f"Invalid compliance clip span: {record}")
    q = np.asarray(arrays['joint_pos'][lo:lo+n+1], dtype=np.float64)
    rp = np.asarray(arrays['root_pos_w'][lo:lo+n], dtype=np.float64)
    rq = np.asarray(arrays['root_quat_w'][lo:lo+n], dtype=np.float64)
    _, j = geometry_jacobians(q[:-1], rp, rq, meta['joint_names'], points, geometry)
    A = j[:, :, :, 6:].transpose(0, 3, 1, 2).reshape(n, q.shape[1], -1)
    dq = np.diff(q, axis=0)
    forces = np.zeros((n, len(points), 3), dtype=np.float32)
    previous = None
    for start in range(0, n, 64):
        end = min(n, start + 64)
        f, _ = solve_sequence(A[start:end], dq[start:end], budget, k=k,
                              slew=150, smooth=.01, previous_force=previous)
        forces[start:end] = f.reshape(end-start, len(points), 3).astype(np.float32)
        previous = f[-1]
    return dict(dataset=record['dataset'], clip=clip,
                motion_id=int(record['source_motion_id']),
                start_step=int(record['source_start_step']),
                frames=n, forces=forces)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT/'cfg/compliance_eval_v1_1.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source_cfg = json.loads((ROOT/'artifacts/offline_force_baseline/train37_budget60/manifest.json').read_text())
    geometry = source_cfg['geometry']
    points = source_cfg['points']
    k = float(source_cfg['K'])
    budget = float(source_cfg['budget_N'])
    manifest = json.loads(a.manifest.read_text())
    # Seeds repeat the same source clip; calculate each clip's prompt once.
    records = {}
    for row in manifest['clips']:
        key = (row['dataset'], int(row['clip']))
        records[key] = row
    tasks = list(records.values())
    solved = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(solve_record, row, points, geometry, k, budget, str(a.output)) for row in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            solved.append(f.result())
            if i % 20 == 0 or i == len(futs):
                print(f'SOLVED {i}/{len(futs)}', flush=True)
    mapping = {}
    for dataset in sorted({x['dataset'] for x in solved}):
        entries = sorted((x for x in solved if x['dataset'] == dataset), key=lambda x:(x['motion_id'],x['start_step']))
        offsets=[]; ids=[]; starts=[]; lengths=[]; chunks=[]; offset=0
        for e in entries:
            ids.append(e['motion_id']); starts.append(e['start_step']); lengths.append(e['frames']); offsets.append(offset)
            chunks.append(e['forces']); offset += e['frames']
        cache_path = a.output / (dataset.replace('/', '__').replace('..','parent') + '.npz')
        np.savez_compressed(cache_path,
            motion_ids=np.asarray(ids,dtype=np.int64),
            start_steps=np.asarray(starts,dtype=np.int64),
            frames=np.asarray(lengths,dtype=np.int64),
            offsets=np.asarray(offsets,dtype=np.int64),
            forces=np.concatenate(chunks,axis=0))
        source = (ROOT/'dataset'/dataset).resolve()
        mapping[dataset] = dict(path=str(cache_path.resolve()),
            dataset_meta_sha256=digest(source/'meta_motion.json'),
            records=len(entries), transitions=int(sum(lengths)))
    out=dict(version='offline-force-prompt-compliance-v1', manifest_sha256=digest(a.manifest),
             source_prompt_cache_manifest_sha256=digest(ROOT/'artifacts/offline_force_baseline/train37_budget60/manifest.json'),
             K=k,budget_N=budget,dt=.02,points=points,geometry_sha256=digest(ROOT/'artifacts/ceer_segmented_jacobian/geometry.json'),
             datasets=mapping)
    (a.output/'cache_map.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({d:{'records':v['records'],'transitions':v['transitions'],'path':v['path']} for d,v in mapping.items()},indent=2))


if __name__ == '__main__':
    main()
