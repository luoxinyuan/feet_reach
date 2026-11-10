"""Distributed one-pass supervised training over sharded low-level rollouts."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16384, help='per GPU')
    p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--validation-samples-per-rank', type=int, default=200000)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from active_adaptation.learning.ppo.common import make_mlp

    local_rank = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank); device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group('nccl')
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.manual_seed(a.seed + rank); rng = np.random.default_rng(a.seed + rank)
    torch.set_num_threads(4)

    manifests = [json.loads(p.read_text()) for p in sorted(a.data_dir.glob('rank*_manifest.json'))]
    if len(manifests) != world:
        raise RuntimeError(f'Expected {world} rollout shards, found {len(manifests)}')
    manifest = manifests[rank]
    stem = a.data_dir / f'rank{rank:02d}'
    n = int(manifest['transitions'])
    force = np.load(stem.with_name(stem.name + '_force_N.npy'), mmap_mode='r')
    delta = np.load(stem.with_name(stem.name + '_delta_q_rad.npy'), mmap_mode='r')
    motion = np.load(stem.with_name(stem.name + '_motion_id.npy'), mmap_mode='r')
    if force.shape != (n, 9) or delta.shape != (n, 29) or motion.shape != (n,):
        raise RuntimeError(f'Invalid shard shapes: {force.shape}, {delta.shape}, {motion.shape}')

    # Keep clips intact across train/validation using stable motion IDs.
    def is_val(mid):
        ids = np.asarray(mid, dtype=np.uint64)
        return ((ids * np.uint64(2654435761) + np.uint64(2246822519)) % np.uint64(5)) == 0

    chunk = 8_000_000
    train_count = 0; val_count = 0
    for start in range(0, n, chunk):
        mask = is_val(motion[start:min(n, start + chunk)])
        val_count += int(mask.sum()); train_count += len(mask) - int(mask.sum())
    counts = torch.tensor([train_count, val_count], device=device, dtype=torch.int64)
    gathered = [torch.zeros_like(counts) for _ in range(world)]
    dist.all_gather(gathered, counts)
    min_train = min(int(x[0].item()) for x in gathered)
    steps = min_train // a.batch_size
    used_count = steps * a.batch_size
    if steps < 1: raise RuntimeError('Not enough training transitions per GPU')

    # Build and shuffle host-side row indices; the 4B-scale tensors stay on disk.
    train_indices = np.empty(train_count, dtype=np.int64); cursor = 0
    val_indices = np.empty(min(val_count, a.validation_samples_per_rank * 4), dtype=np.int64)
    val_cursor = 0
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        ids = np.arange(start, end, dtype=np.int64)
        vm = is_val(motion[start:end])
        tr = ids[~vm]; train_indices[cursor:cursor + len(tr)] = tr; cursor += len(tr)
        va = ids[vm]
        if val_cursor < len(val_indices):
            take = min(len(va), len(val_indices) - val_cursor)
            val_indices[val_cursor:val_cursor + take] = va[:take]; val_cursor += take
    assert cursor == train_count
    train_indices = train_indices[:used_count]
    rng.shuffle(train_indices)
    val_indices = val_indices[:val_cursor]
    if len(val_indices) > a.validation_samples_per_rank:
        val_indices = rng.choice(val_indices, a.validation_samples_per_rank, replace=False)

    def as_f32(mm, ids):
        return torch.from_numpy(np.asarray(mm[ids], dtype=np.float32).copy()).to(device)

    # Global, train-only normalization statistics from a random sample.
    stat_n = min(1_000_000, used_count)
    stat_idx = rng.choice(train_indices, stat_n, replace=False)
    sx = as_f32(force, stat_idx); sy = as_f32(delta, stat_idx)
    stats = torch.cat((sx.sum(0), sx.square().sum(0), sy.square().sum(0),
                       torch.tensor([float(len(sx))], device=device)))
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    x_mean = stats[:9] / stats[-1]
    x_std = (stats[9:18] / stats[-1] - x_mean.square()).clamp_min(1e-6).sqrt()
    y_scale = (stats[18:47] / stats[-1]).clamp_min(1e-8).sqrt()

    model = nn.Sequential(make_mlp([512, 256], norm='before'), nn.Linear(256, 29)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train(); loss_sum = torch.zeros((), device=device)
    # Shuffle once, then consume every selected train transition exactly once.
    bytes_per_batch = a.batch_size
    for step in range(steps):
        ids = train_indices[step * bytes_per_batch:(step + 1) * bytes_per_batch]
        x = (as_f32(force, ids) - x_mean) / x_std
        y = as_f32(delta, ids) / y_scale
        loss = F.smooth_l1_loss(model(x), y, beta=1.0)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        loss_sum += loss.detach()
        if rank == 0 and (step + 1) % 1000 == 0:
            print(f'train_step={step+1}/{steps} loss={loss.item():.6f}', flush=True)

    model.eval(); stats_eval = torch.zeros(1 + 4 * 29 + 1, device=device)
    with torch.no_grad():
        for start in range(0, len(val_indices), a.batch_size):
            ids = val_indices[start:start + a.batch_size]
            x_raw = as_f32(force, ids); y_raw = as_f32(delta, ids)
            pred = model((x_raw - x_mean) / x_std) * y_scale
            err = pred - y_raw
            moving = y_raw.abs() > .01
            cos = F.cosine_similarity(y_raw, pred, dim=-1)
            stats_eval[0] += len(ids)
            stats_eval[1:30] += err.abs().sum(0)
            stats_eval[30:59] += err.square().sum(0)
            stats_eval[59:88] += y_raw.sum(0)
            stats_eval[88:117] += y_raw.square().sum(0)
            stats_eval[117] += cos.sum()
    dist.all_reduce(stats_eval, op=dist.ReduceOp.SUM)
    denom = stats_eval[0].clamp_min(1.)
    mae = stats_eval[1:30] / denom
    mse = stats_eval[30:59] / denom
    variance = stats_eval[88:117] / denom - (stats_eval[59:88] / denom).square()
    r2_joint = 1. - mse / variance.clamp_min(1e-12)
    mae_mean = float(mae.mean().item()); rmse = float(mse.mean().sqrt().item())
    r2_macro = float(r2_joint.mean().item())
    val_count_global = int(stats_eval[0].item())
    cos_mean = float((stats_eval[117] / denom).item())
    train_loss = torch.tensor([float(loss_sum.item()), float(steps)], device=device)
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        report = {'samples_total': int(sum(m['transitions'] for m in manifests)),
            'train_samples_used': int(used_count * world), 'validation_samples_evaluated': val_count_global,
            'optimizer_steps_global': int(steps), 'per_gpu_batch_size': a.batch_size,
            'force_dim': 9, 'output_dim': 29, 'points': manifests[0]['force_points'],
            'joint_names': manifests[0]['joint_names'], 'architecture': '9 -> MLP(512,256,LayerNorm-before) -> 29; output scaled by per-joint delta-q std',
            'validation_mae_rad': mae_mean, 'validation_rmse_rad': rmse,
            'validation_macro_joint_r2': r2_macro, 'validation_mean_cosine': cos_mean,
            'input_mean_N': x_mean.cpu().tolist(), 'input_std_N': x_std.cpu().tolist(),
            'output_scale_rad': y_scale.cpu().tolist(), 'seed': a.seed,
            'data_dir': str(a.data_dir.resolve())}
        torch.save({'model': model.module.state_dict(), 'input_mean_N': x_mean.cpu(),
            'input_std_N': x_std.cpu(), 'output_scale_rad': y_scale.cpu(),
            'point_names': manifests[0]['force_points'], 'joint_names': manifests[0]['joint_names'],
            'architecture': report['architecture'], 'seed': a.seed}, a.output_dir / 'residual_teacher.pt')
        (a.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        (a.output_dir / 'summary.md').write_text(
            '# Supervised 3-point force-to-delta-q residual teacher\n\n'
            f"Trained on {report['train_samples_used']:,} rollouts, with {val_count_global:,} held-out transitions grouped by motion ID.\n\n"
            '| Validation MAE (rad) | RMSE (rad) | Macro joint R² | Mean Δq cosine |\n'
            '|---:|---:|---:|---:|\n'
            f"| {mae_mean:.5f} | {rmse:.5f} | {r2_macro:.3f} | {cos_mean:.3f} |\n")
        print('SUPERVISED_TRAINING_COMPLETE ' + json.dumps(report), flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__': main()
