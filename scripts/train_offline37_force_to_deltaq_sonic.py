"""Train a full-corpus 37-point force -> next-step joint-delta model."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from active_adaptation.learning.ppo.common import make_mlp


def quat_apply_inverse(quat, vec):
    xyz = quat[..., 1:]
    w = quat[..., :1]
    t = torch.cross(xyz, vec, dim=-1) * 2
    return vec - w * t + torch.cross(xyz, t, dim=-1)


def build_model(input_dim=111):
    model = nn.Sequential(make_mlp([512, 256], norm='before'), nn.Linear(256, 29))
    with torch.no_grad():
        model(torch.zeros(1, input_dim))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=4096, help='per GPU')
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--lr', type=float, default=5e-4)
    a = p.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    torch.set_num_threads(4)

    manifest_path = a.cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    points = manifest['points']
    point_names = [x['name'] for x in points]
    joint_names = manifest['joint_names']
    if len(points) != 37 or len(joint_names) != 29:
        raise ValueError(f'Expected 37 force points / 29 joints, got {len(points)} / {len(joint_names)}')
    force_path = a.cache / 'forces.npy'
    force = np.load(force_path, mmap_mode='r')
    expected_transitions = int(manifest['summary']['transitions'])
    if force.shape != (int(manifest['frames']), 37, 3):
        raise ValueError(f'Unexpected force cache shape {force.shape}')

    datasets = []
    for i, path in enumerate(manifest['dataset_paths']):
        root = Path(path)
        tdmeta = json.loads((root / '_tensordict' / 'meta.json').read_text())
        shape = tdmeta['joint_pos']['shape']
        quat_shape = tdmeta['root_quat_w']['shape']
        q = np.memmap(root / '_tensordict' / 'joint_pos.memmap', mode='r', dtype=np.float16, shape=tuple(shape))
        quat = np.memmap(root / '_tensordict' / 'root_quat_w.memmap', mode='r', dtype=np.float16, shape=tuple(quat_shape))
        meta = json.loads((root / 'meta_motion.json').read_text())
        if meta['joint_names'] != joint_names:
            raise ValueError(f'Joint ordering mismatch in {root}')
        datasets.append((q, quat))

    # Greedily balance whole clips across DDP ranks; no clip is split.
    assignments = [[] for _ in range(world)]
    loads = [0] * world
    clips = manifest['clips']
    for row in sorted(clips, key=lambda c: int(c['frames']), reverse=True):
        r = min(range(world), key=lambda j: loads[j])
        count = int(row['frames']) - 1
        if count <= 0:
            continue
        assignments[r].append(row)
        loads[r] += count
    local_clips = assignments[rank]
    local_count = loads[rank]
    all_counts = [torch.zeros((), device=device, dtype=torch.long) for _ in range(world)]
    local_count_t = torch.tensor(local_count, device=device, dtype=torch.long)
    dist.all_gather(all_counts, local_count_t)
    steps_per_epoch = min(int(x.item()) for x in all_counts) // a.batch_size
    used_per_rank = steps_per_epoch * a.batch_size
    if steps_per_epoch < 1:
        raise RuntimeError('Not enough transitions per GPU')

    # Concatenate only index metadata. Large arrays remain memory mapped.
    starts = np.cumsum([0] + [int(c['frames']) - 1 for c in local_clips[:-1]], dtype=np.int64)
    ends = starts + np.asarray([int(c['frames']) - 1 for c in local_clips], dtype=np.int64)
    data_ids = np.asarray([int(c['source_index']) for c in local_clips], dtype=np.int64)
    q_starts = np.asarray([int(c['start']) for c in local_clips], dtype=np.int64)
    force_starts = np.asarray([int(c['force_offset']) for c in local_clips], dtype=np.int64)

    def batch_from_ids(ids):
        ids = np.asarray(ids, dtype=np.int64)
        clip_idx = np.searchsorted(ends, ids, side='right')
        x_world = np.empty((len(ids), 37, 3), dtype=np.float32)
        roots = np.empty((len(ids), 4), dtype=np.float32)
        y = np.empty((len(ids), 29), dtype=np.float32)
        local_step = ids - starts[clip_idx]
        force_idx = force_starts[clip_idx] + local_step
        q_idx = q_starts[clip_idx] + local_step
        for dsid in np.unique(data_ids[clip_idx]):
            mask = data_ids[clip_idx] == dsid
            q, quat = datasets[dsid]
            x_world[mask] = force[force_idx[mask]]
            roots[mask] = quat[q_idx[mask]]
            y[mask] = q[q_idx[mask] + 1].astype(np.float32) - q[q_idx[mask]].astype(np.float32)
        xw = torch.from_numpy(x_world).to(device)
        qr = torch.from_numpy(roots).to(device)
        x = quat_apply_inverse(qr[:, None, :].expand(-1, 37, -1), xw).reshape(len(ids), 111)
        return x, torch.from_numpy(y).to(device)

    # Compute force normalization and per-joint target scale from the full corpus.
    rng = np.random.default_rng(a.seed + rank)
    stat_ids = rng.integers(0, local_count, size=min(200_000, local_count), dtype=np.int64)
    sx, sy = batch_from_ids(stat_ids)
    stat = torch.cat((sx.sum(0), sx.square().sum(0), sy.square().sum(0),
                      torch.tensor([float(len(sx))], device=device)))
    dist.all_reduce(stat)
    x_mean = stat[:111] / stat[-1]
    x_std = (stat[111:222] / stat[-1] - x_mean.square()).clamp_min(1e-6).sqrt().clamp_min(1e-3)
    y_scale = (stat[222:251] / stat[-1]).clamp_min(1e-8).sqrt().clamp_min(1e-4)

    model = build_model(111).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    start_time = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        order = np.random.default_rng(a.seed + epoch).permutation(local_count)[:used_per_rank]
        loss_total = torch.zeros((), device=device)
        for step in range(steps_per_epoch):
            ids = order[step * a.batch_size:(step + 1) * a.batch_size]
            x, y = batch_from_ids(ids)
            pred = model((x - x_mean) / x_std)
            loss = F.smooth_l1_loss(pred, y / y_scale, beta=1.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_total += loss.detach()
        dist.all_reduce(loss_total)
        if rank == 0:
            print(json.dumps({'epoch': epoch + 1, 'epochs': a.epochs,
                              'train_loss_normalized': float(loss_total.item() / world / steps_per_epoch),
                              'steps_per_epoch': steps_per_epoch,
                              'elapsed_s': time.monotonic() - start_time}), flush=True)

    # A small random in-corpus diagnostic; the authoritative generalization test is the fixed benchmark.
    model.eval()
    eval_ids = rng.integers(0, local_count, size=min(100_000, local_count), dtype=np.int64)
    abs_sum = torch.zeros((), device=device)
    sq_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    with torch.no_grad():
        for start in range(0, len(eval_ids), a.batch_size):
            x, y = batch_from_ids(eval_ids[start:start + a.batch_size])
            pred = model((x - x_mean) / x_std) * y_scale
            err = pred - y
            abs_sum += err.abs().sum()
            sq_sum += err.square().sum()
            count += err.numel()
    stats_eval = torch.stack([abs_sum, sq_sum, count])
    dist.all_reduce(stats_eval)
    if rank == 0:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        mae = float((stats_eval[0] / stats_eval[2]).item())
        rmse = float((stats_eval[1] / stats_eval[2]).sqrt().item())
        report = {
            'source_cache': str(a.cache.resolve()),
            'source_manifest_sha256': __import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest(),
            'samples_total': expected_transitions,
            'clips_total': len(clips),
            'samples_per_rank': loads,
            'train_epochs': a.epochs,
            'batch_size_per_gpu': a.batch_size,
            'optimizer_steps_per_rank': steps_per_epoch * a.epochs,
            'force_dim': 111,
            'output_dim': 29,
            'force_points': point_names,
            'joint_names': joint_names,
            'force_coordinates': 'root frame, N; matches OfflineForcePromptCommand observation before /30 normalization',
            'target': 'next-frame joint_pos[t+1] - joint_pos[t], radians',
            'architecture': '111 -> MLP(512,256,LayerNorm-before) -> 29; per-joint RMS output scale',
            'in_corpus_diagnostic_mae_rad': mae,
            'in_corpus_diagnostic_rmse_rad': rmse,
            'elapsed_s': time.monotonic() - start_time,
            'seed': a.seed,
        }
        a.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': model.module.state_dict(),
            'input_mean_N': x_mean.cpu(),
            'input_std_N': x_std.cpu(),
            'output_scale_rad': y_scale.cpu(),
            'point_names': point_names,
            'joint_names': joint_names,
            'force_input_dim': 111,
            'history_offsets_frames': [0],
            'force_coordinates': 'root frame',
            'architecture': report['architecture'],
            'seed': a.seed,
        }, a.output_dir / 'residual_teacher.pt')
        (a.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        (a.output_dir / 'summary.md').write_text(
            '# 37-point force to SONIC delta-q model\n\n'
            f"Trained on all {expected_transitions:,} cached transitions from {len(clips):,} clips for {a.epochs} epochs.\n\n"
            'Input: current 37-point force tokens in the root frame (111 values). Output: next-frame 29-joint delta-q in radians.\n\n'
            f'Random in-corpus diagnostic MAE: {mae:.6f} rad; RMSE: {rmse:.6f} rad. The compliance-v1.1 rollout is the generalization evaluation.\n')
        print('SUPERVISED_37PT_TRAINING_COMPLETE ' + json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
