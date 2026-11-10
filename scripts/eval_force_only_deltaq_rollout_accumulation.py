"""Integrate force-only delta-q predictions and measure trajectory error growth."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_offline37_force_to_deltaq_sonic import build_model, quat_apply_inverse


ROOT = Path(__file__).resolve().parents[1]


def error_stats(err):
    return {'mae_rad': float(np.abs(err).mean()), 'rmse_rad': float(np.sqrt(np.square(err).mean()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT / 'cfg/compliance_eval_v1_1.json')
    p.add_argument('--force-cache-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='JSON report; compressed trajectory NPZ is saved beside it')
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

    gt_sequences, pred_sequences, clips_detail = [], [], []
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
            cache_motion_ids, cache_starts = cache['motion_ids'], cache['start_steps']
            cache_frames, cache_offsets, cache_forces = cache['frames'], cache['offsets'], cache['forces']
            unique_ids, starts, counts = np.unique(motion_ids, return_index=True, return_counts=True)
            motion_row = {int(mid): (int(start), int(count)) for mid, start, count in zip(unique_ids, starts, counts)}

            for row in dataset_rows:
                mid, source_start, frames = (int(row['source_motion_id']), int(row['source_start_step']), int(row['frames']))
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

                q_gt = np.asarray(q[q_start:q_start + frames], dtype=np.float32)
                root_quat = torch.from_numpy(np.asarray(quat[q_start:q_start + frames - 1], dtype=np.float32))
                force = torch.from_numpy(force_world)
                force_root = quat_apply_inverse(root_quat[:, None, :].expand(-1, 37, -1), force).reshape(-1, 111)
                delta_pred = (model((force_root - x_mean) / x_std) * y_scale).numpy()

                q_pred = np.empty_like(q_gt, dtype=np.float32)
                q_pred[0] = q_gt[0]  # initialize from ground-truth starting pose
                q_pred[1:] = q_pred[0] + np.cumsum(delta_pred, axis=0)
                err = q_pred[1:] - q_gt[1:]
                clip = {
                    'dataset': dataset_key, 'motion_id': mid, 'source_start_step': source_start,
                    'frames': frames, 'category': row['category'],
                    'mean_trajectory_mae_rad': float(np.abs(err).mean()),
                    'mean_trajectory_rmse_rad': float(np.sqrt(np.square(err).mean())),
                    'endpoint_mae_rad': float(np.abs(err[-1]).mean()),
                    'endpoint_rmse_rad': float(np.sqrt(np.square(err[-1]).mean())),
                    'sequence_offset': int(sum(len(x) for x in gt_sequences)),
                }
                clips_detail.append(clip)
                gt_sequences.append(q_gt)
                pred_sequences.append(q_pred)

    # Error at each elapsed step, excluding the exactly initialized t=0 frame.
    horizons = {}
    for k in [1, 5, 10, 25, 50, 100, 200, 300, 500]:
        eligible = [(gt[k], pred[k]) for gt, pred in zip(gt_sequences, pred_sequences) if len(gt) > k]
        if eligible:
            err = np.concatenate([pred - gt for gt, pred in eligible], axis=0)
            horizons[str(k)] = {'clips': len(eligible), **error_stats(err)}
    segments = {}
    for lo, hi in [(1, 10), (10, 50), (50, 100), (100, 200), (200, 300), (300, 500)]:
        errors = []
        for gt, pred in zip(gt_sequences, pred_sequences):
            end = min(hi, len(gt) - 1)
            if end >= lo:
                errors.append((pred[lo:end + 1] - gt[lo:end + 1]).reshape(-1, 29))
        if errors:
            segments[f'{lo}-{hi}'] = error_stats(np.concatenate(errors, axis=0))

    all_err = np.concatenate([pseq[1:] - gseq[1:] for gseq, pseq in zip(gt_sequences, pred_sequences)], axis=0)
    endpoint_err = np.stack([pseq[-1] - gseq[-1] for gseq, pseq in zip(gt_sequences, pred_sequences)])
    result = {
        'model_type': 'force_to_delta_q_integrated_rollout', 'checkpoint': str(a.checkpoint.resolve()),
        'split': a.split, 'input_contract': 'aligned ground-truth 37-point force-token sequence only; no q input after initialization',
        'integration_contract': 'q_pred[0] = ground-truth q[0]; q_pred[t+1] = q_pred[t] + predicted_delta_q[t]',
        'clips': len(clips_detail), 'transitions': int(sum(c['frames'] - 1 for c in clips_detail)),
        'trajectory_mae_rad': float(np.abs(all_err).mean()),
        'trajectory_rmse_rad': float(np.sqrt(np.square(all_err).mean())),
        'endpoint_mae_rad': float(np.abs(endpoint_err).mean()),
        'endpoint_rmse_rad': float(np.sqrt(np.square(endpoint_err).mean())),
        'mean_clip_trajectory_mae_rad': float(np.mean([c['mean_trajectory_mae_rad'] for c in clips_detail])),
        'mean_clip_endpoint_mae_rad': float(np.mean([c['endpoint_mae_rad'] for c in clips_detail])),
        'horizon_errors': horizons, 'elapsed_step_segments': segments,
        'per_joint_endpoint_mae_rad': {name: float(v) for name, v in zip(joint_names, np.abs(endpoint_err).mean(0))},
        'caveat': 'Diagnostic open-loop integration: force input remains the held-out aligned force sequence and the initial pose is ground truth. It measures accumulated reference-q error, not robot physics or closed-loop SONIC tracking.',
        'clips_detail': clips_detail,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    seq_path = a.output.with_suffix('.npz')
    offsets = np.cumsum([0] + [len(x) for x in gt_sequences], dtype=np.int64)
    np.savez_compressed(seq_path,
                        q_ground_truth=np.concatenate(gt_sequences, axis=0),
                        q_integrated_prediction=np.concatenate(pred_sequences, axis=0),
                        sequence_offsets=offsets,
                        joint_names=np.asarray(joint_names),
                        clip_keys=np.asarray([f"{c['dataset']}|{c['motion_id']}|{c['source_start_step']}" for c in clips_detail]))
    result['trajectory_npz'] = str(seq_path.resolve())
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'clips_detail'}, indent=2))


if __name__ == '__main__':
    main()
