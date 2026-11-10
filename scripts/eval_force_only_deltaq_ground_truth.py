"""Teacher-forced ground-truth audit for the 37-point force -> delta-q model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_offline37_force_to_deltaq_sonic import build_model, quat_apply_inverse


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
    model = build_model(input_dim=111).eval()
    model.load_state_dict(blob['model'], strict=True)
    x_mean = torch.as_tensor(blob['input_mean_N'], dtype=torch.float32)
    x_std = torch.as_tensor(blob['input_std_N'], dtype=torch.float32)
    y_scale = torch.as_tensor(blob['output_scale_rad'], dtype=torch.float32)

    protocol = json.loads(a.manifest.read_text())
    cache_map = json.loads(a.force_cache_map.read_text())['datasets']
    rows = [r for r in protocol['clips'] if r['split'] == a.split]
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row['dataset'], []).append(row)

    abs_sum = np.zeros(29, dtype=np.float64)
    sq_sum = np.zeros(29, dtype=np.float64)
    target_sq_sum = np.zeros(29, dtype=np.float64)
    pred_sq_sum = np.zeros(29, dtype=np.float64)
    target_sum = np.zeros(29, dtype=np.float64)
    per_clip = []
    total_transitions = 0
    joint_names = list(blob['joint_names'])

    with torch.inference_mode():
        for dataset_key, dataset_rows in by_dataset.items():
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
            if dataset_meta['joint_names'] != joint_names:
                raise ValueError(f'Joint ordering mismatch: {dataset_root}')

            cache_entry = cache_map[dataset_key]
            cache = np.load(cache_entry['path'], allow_pickle=False)
            cache_motion_ids = cache['motion_ids']
            cache_starts = cache['start_steps']
            cache_frames = cache['frames']
            cache_offsets = cache['offsets']
            cache_forces = cache['forces']

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
                expected_steps = np.arange(source_start, source_start + frames, dtype=np.int32)
                if q_start_local + frames > motion_count or not np.array_equal(
                        local_steps[q_start_local:q_start_local + frames], expected_steps):
                    raise ValueError(f'Clip frames do not align for motion={mid}, start={source_start}')
                q_start = motion_start + q_start_local

                candidates = np.flatnonzero(cache_motion_ids == mid)
                selected = [int(i) for i in candidates
                            if int(cache_starts[i]) <= source_start
                            and source_start + frames - 1 <= int(cache_starts[i] + cache_frames[i])]
                if len(selected) != 1:
                    raise ValueError(f'Expected one covering force record for motion={mid}, got {selected}')
                ci = selected[0]
                force_start = int(cache_offsets[ci] + source_start - cache_starts[ci])
                force_world = np.asarray(cache_forces[force_start:force_start + frames - 1], dtype=np.float32)
                if force_world.shape != (frames - 1, 37, 3):
                    raise ValueError(f'Force clip shape mismatch: {force_world.shape}')

                q_now = np.asarray(q[q_start:q_start + frames - 1], dtype=np.float32)
                q_next = np.asarray(q[q_start + 1:q_start + frames], dtype=np.float32)
                target_delta = q_next - q_now
                root_quat = torch.from_numpy(np.asarray(quat[q_start:q_start + frames - 1], dtype=np.float32))
                force = torch.from_numpy(force_world)
                force_root = quat_apply_inverse(root_quat[:, None, :].expand(-1, 37, -1), force).reshape(-1, 111)
                pred_delta = (model((force_root - x_mean) / x_std) * y_scale).numpy()
                err = pred_delta - target_delta

                abs_sum += np.abs(err).sum(0)
                sq_sum += np.square(err).sum(0)
                target_sq_sum += np.square(target_delta).sum(0)
                pred_sq_sum += np.square(pred_delta).sum(0)
                target_sum += target_delta.sum(0)
                total_transitions += len(err)
                per_clip.append({
                    'dataset': dataset_key, 'motion_id': mid, 'source_start_step': source_start,
                    'frames': frames, 'category': row['category'],
                    'mae_rad': float(np.abs(err).mean()),
                    'rmse_rad': float(np.sqrt(np.square(err).mean())),
                    'target_delta_rms_rad': float(np.sqrt(np.square(target_delta).mean())),
                    'prediction_delta_rms_rad': float(np.sqrt(np.square(pred_delta).mean())),
                })

    n_values = total_transitions * 29
    per_joint_mae = abs_sum / total_transitions
    per_joint_rmse = np.sqrt(sq_sum / total_transitions)
    target_mse = np.maximum(target_sq_sum / total_transitions - (target_sum / total_transitions) ** 2, 1e-12)
    per_joint_r2 = 1.0 - sq_sum / total_transitions / target_mse
    category_summary = {}
    for category in sorted({r['category'] for r in per_clip}):
        group = [r for r in per_clip if r['category'] == category]
        category_summary[category] = {
            'clips': len(group),
            'mean_clip_mae_rad': float(np.mean([r['mae_rad'] for r in group])),
            'mean_clip_rmse_rad': float(np.mean([r['rmse_rad'] for r in group])),
        }

    result = {
        'model_type': 'force_to_delta_q', 'checkpoint': str(a.checkpoint.resolve()), 'split': a.split,
        'input_contract': 'same-frame 37-point force tokens, transformed to root frame; no joint-angle input',
        'target_contract': 'predicted delta_q[t] compared with ground-truth q[t+1] - q[t]',
        'clips': len(per_clip), 'transitions': total_transitions, 'joint_values': n_values,
        'transition_weighted_mae_rad': float(abs_sum.sum() / n_values),
        'transition_weighted_rmse_rad': float(np.sqrt(sq_sum.sum() / n_values)),
        'target_delta_rms_rad': float(np.sqrt(target_sq_sum.sum() / n_values)),
        'prediction_delta_rms_rad': float(np.sqrt(pred_sq_sum.sum() / n_values)),
        'normalized_rmse_by_target_std': float(np.sqrt(np.mean(sq_sum / total_transitions / target_mse))),
        'macro_joint_r2': float(np.mean(per_joint_r2)),
        'mean_clip_mae_rad': float(np.mean([r['mae_rad'] for r in per_clip])),
        'mean_clip_rmse_rad': float(np.mean([r['rmse_rad'] for r in per_clip])),
        'per_joint_mae_rad': {name: float(v) for name, v in zip(joint_names, per_joint_mae)},
        'per_joint_rmse_rad': {name: float(v) for name, v in zip(joint_names, per_joint_rmse)},
        'per_joint_r2': {name: float(v) for name, v in zip(joint_names, per_joint_r2)},
        'category_summary': category_summary,
        'caveat': 'Teacher-forced ground-truth trajectory metric; force-only inference, no measured q is provided. This measures one-step delta prediction, not accumulated closed-loop rollout drift.',
        'clips_detail': per_clip,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('clips_detail', 'per_joint_mae_rad', 'per_joint_rmse_rad', 'per_joint_r2')}, indent=2))


if __name__ == '__main__':
    main()
