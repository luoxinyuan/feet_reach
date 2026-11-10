"""Teacher-forced validation of the force/current-q -> next-q model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_force_to_deltaq_residual import build_model
from scripts.train_offline37_force_current_q_to_next_q_sonic import quat_apply_inverse


ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT / 'cfg/compliance_eval_v1_1.json')
    p.add_argument('--force-cache-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'train_diagnostic'], default='validation')
    a = p.parse_args()

    blob = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if blob.get('model_type') != 'force_current_q_to_next_q':
        raise ValueError(f"Unexpected model_type: {blob.get('model_type')}")
    model = build_model(input_dim=140).eval()
    model.load_state_dict(blob['model'], strict=True)
    x_mean = torch.as_tensor(blob['input_mean'], dtype=torch.float32)
    x_std = torch.as_tensor(blob['input_std'], dtype=torch.float32)
    y_mean = torch.as_tensor(blob['output_mean_rad'], dtype=torch.float32)
    y_std = torch.as_tensor(blob['output_std_rad'], dtype=torch.float32)

    protocol = json.loads(a.manifest.read_text())
    cache_map = json.loads(a.force_cache_map.read_text())['datasets']
    rows = [r for r in protocol['clips'] if r['split'] == a.split]
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row['dataset'], []).append(row)

    all_abs, all_sq, per_clip = [], [], []
    per_joint_abs = np.zeros(29, dtype=np.float64)
    per_joint_sq = np.zeros(29, dtype=np.float64)
    total_values = 0
    model_joint_names = list(blob['joint_names'])
    dataset_joint_names = None

    with torch.inference_mode():
        for dataset_key, dataset_rows in by_dataset.items():
            # Validation dataset keys are relative to the project's dataset directory.
            if dataset_key.startswith('../'):
                dataset_root = (ROOT / 'dataset' / dataset_key).resolve()
            else:
                dataset_root = (ROOT / 'dataset' / dataset_key).resolve()
            tdroot = dataset_root / '_tensordict'
            meta = json.loads((tdroot / 'meta.json').read_text())
            motion_ids = np.memmap(tdroot / 'motion_id.memmap', mode='r', dtype=np.int32,
                                   shape=tuple(meta['motion_id']['shape']))
            steps = np.memmap(tdroot / 'step.memmap', mode='r', dtype=np.int32,
                              shape=tuple(meta['step']['shape']))
            q = np.memmap(tdroot / 'joint_pos.memmap', mode='r', dtype=np.float16,
                          shape=tuple(meta['joint_pos']['shape']))
            quat = np.memmap(tdroot / 'root_quat_w.memmap', mode='r', dtype=np.float16,
                             shape=tuple(meta['root_quat_w']['shape']))
            dataset_meta = json.loads((dataset_root / 'meta_motion.json').read_text())
            if dataset_joint_names is None:
                dataset_joint_names = dataset_meta['joint_names']
            if dataset_meta['joint_names'] != model_joint_names:
                raise ValueError(f'Joint order mismatch: {dataset_root}')

            cache_entry = cache_map[dataset_key]
            cache = np.load(cache_entry['path'], allow_pickle=False)
            cache_motion_ids = cache['motion_ids']
            cache_starts = cache['start_steps']
            cache_frames = cache['frames']
            cache_offsets = cache['offsets']
            cache_forces = cache['forces']

            # Source motions occupy contiguous rows; index each motion once.
            unique_ids, starts, counts = np.unique(motion_ids, return_index=True, return_counts=True)
            motion_row = {int(mid): (int(start), int(count)) for mid, start, count in zip(unique_ids, starts, counts)}

            for row in dataset_rows:
                mid = int(row['source_motion_id'])
                source_start = int(row['source_start_step'])
                frames = int(row['frames'])
                if mid not in motion_row:
                    raise ValueError(f'Motion {mid} missing in {dataset_root}')
                motion_start, motion_count = motion_row[mid]
                local_steps = steps[motion_start:motion_start + motion_count]
                q_start_local = int(np.searchsorted(local_steps, source_start))
                if q_start_local + frames > motion_count or not np.array_equal(
                        local_steps[q_start_local:q_start_local + frames],
                        np.arange(source_start, source_start + frames, dtype=np.int32)):
                    raise ValueError(f'Clip frames do not align for motion={mid}, start={source_start}, dataset={dataset_key}')
                q_start = motion_start + q_start_local

                candidates = np.flatnonzero(cache_motion_ids == mid)
                selected = [int(i) for i in candidates
                            if int(cache_starts[i]) <= source_start
                            and source_start + frames - 1 <= int(cache_starts[i] + cache_frames[i])]
                if len(selected) != 1:
                    raise ValueError(f'Expected one covering force record for motion={mid}, start={source_start}, got {selected}')
                ci = selected[0]
                force_start = int(cache_offsets[ci] + source_start - cache_starts[ci])
                force_world = np.asarray(cache_forces[force_start:force_start + frames - 1], dtype=np.float32)
                if force_world.shape != (frames - 1, 37, 3):
                    raise ValueError(f'Force clip shape mismatch: {force_world.shape}')

                q_now = np.asarray(q[q_start:q_start + frames - 1], dtype=np.float32)
                q_next = np.asarray(q[q_start + 1:q_start + frames], dtype=np.float32)
                root_quat = np.asarray(quat[q_start:q_start + frames - 1], dtype=np.float32)
                f = torch.from_numpy(force_world)
                r = torch.from_numpy(root_quat)
                force_root = quat_apply_inverse(r[:, None, :].expand(-1, 37, -1), f).reshape(-1, 111)
                features = torch.cat((force_root, torch.from_numpy(q_now)), dim=-1)
                pred = model((features - x_mean) / x_std) * y_std + y_mean
                err = pred.numpy() - q_next
                abs_err = np.abs(err)
                sq_err = np.square(err)
                mae = float(abs_err.mean())
                rmse = float(np.sqrt(sq_err.mean()))
                per_clip.append({'dataset': dataset_key, 'motion_id': mid,
                                 'source_start_step': source_start, 'frames': frames,
                                 'category': row['category'], 'mae_rad': mae, 'rmse_rad': rmse})
                per_joint_abs += abs_err.sum(0)
                per_joint_sq += sq_err.sum(0)
                total_values += err.size
                all_abs.append(abs_err.reshape(-1))
                all_sq.append(sq_err.reshape(-1))

    abs_flat = np.concatenate(all_abs)
    sq_flat = np.concatenate(all_sq)
    category_summary = {}
    for category in sorted({r['category'] for r in per_clip}):
        group = [r for r in per_clip if r['category'] == category]
        category_summary[category] = {
            'clips': len(group),
            'mean_clip_mae_rad': float(np.mean([r['mae_rad'] for r in group])),
            'mean_clip_rmse_rad': float(np.mean([r['rmse_rad'] for r in group])),
        }
    result = {
        'model_type': blob['model_type'], 'split': a.split,
        'input_contract': 'ground-truth reference q[t] plus same-frame 37-point force tokens, root frame',
        'target_contract': 'predicted absolute q[t+1] compared with ground-truth reference q[t+1]',
        'clips': len(per_clip), 'transitions': int(sum(r['frames'] - 1 for r in per_clip)),
        'joint_values': int(total_values),
        'transition_weighted_mae_rad': float(abs_flat.mean()),
        'transition_weighted_rmse_rad': float(np.sqrt(sq_flat.mean())),
        'mean_clip_mae_rad': float(np.mean([r['mae_rad'] for r in per_clip])),
        'mean_clip_rmse_rad': float(np.mean([r['rmse_rad'] for r in per_clip])),
        'per_joint_mae_rad': {name: float(v) for name, v in zip(model_joint_names, per_joint_abs / (total_values / 29))},
        'category_summary': category_summary,
        'caveat': 'Teacher-forced reference-state metric: uses q[t] from the reference trajectory, not the robot measured q during closed-loop rollout.',
        'clips_detail': per_clip,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'clips_detail'}, indent=2))


if __name__ == '__main__':
    main()
