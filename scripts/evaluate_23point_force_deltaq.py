"""Held-out-motion probe for whether dense 23-point prompts encode joint delta-q."""
import os
os.environ.setdefault('MEMPATH', os.path.abspath('dataset'))
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from active_adaptation.utils.motion import MotionDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/passive_whole_body_full'
CACHE = ROOT / 'artifacts/ceer_dense_link/full23'
OUT = ROOT / 'outputs/20261103-force-prompt-decodability/23point'
SEED = 230023
FRAMES_PER_CLIP = 64
torch.set_num_threads(8)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def sample_split(split):
    ds = MotionDataset.create_from_path_lazy(f'passive_whole_body_full/{split}')
    forces = np.load(CACHE / split / 'forces.npy', mmap_mode='r')
    valid = np.load(CACHE / split / 'valid_next.npy', mmap_mode='r')
    if len(forces) != len(ds.data):
        raise ValueError(f'{split}: force cache and dataset lengths differ')
    q = ds.data.joint_pos
    mids = ds.data.motion_id
    starts, ends = ds.starts.numpy(), ds.ends.numpy()
    xs_force, xs_state, ys, motion_ids = [], [], [], []
    for lo, hi in zip(starts, ends):
        hi = int(hi); lo = int(lo)
        candidates = np.flatnonzero(valid[lo:hi-1]) + lo
        if not len(candidates):
            continue
        count = min(FRAMES_PER_CLIP, len(candidates))
        chosen = np.sort(rng.choice(candidates, size=count, replace=False))
        # Cache[t] is exactly the prompt built from reference body positions t -> t+1.
        # Label is the corresponding dataset joint_pos[t+1] - joint_pos[t].
        xs_force.append(np.asarray(forces[chosen], dtype=np.float32).reshape(count, -1))
        q0 = q[torch.from_numpy(chosen)].float().numpy()
        q1 = q[torch.from_numpy(chosen + 1)].float().numpy()
        xs_state.append(q0)
        ys.append(q1 - q0)
        motion_ids.append(np.full(count, len(motion_ids), dtype=np.int32))
    return (np.concatenate(xs_force), np.concatenate(xs_state), np.concatenate(ys),
            np.concatenate(motion_ids))


def metrics(y, pred):
    std = np.maximum(y.std(0), 1e-5)
    residual = y - pred
    moving = np.abs(y) > .01
    cos = np.sum(y * pred, axis=1) / np.maximum(
        np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1), 1e-9)
    return {
        'nrmse_per_joint_std': float(np.sqrt(np.mean((residual / std) ** 2))),
        'mae_rad': float(np.mean(np.abs(residual))),
        'r2_global': float(1 - np.sum(residual**2) / max(np.sum((y-y.mean(0))**2), 1e-12)),
        'r2_macro_joint': float(np.mean(1 - np.sum(residual**2, axis=0) /
                                        np.maximum(np.sum((y-y.mean(0))**2, axis=0), 1e-12))),
        'moving_joint_sign_accuracy': float(np.mean(np.sign(y[moving]) == np.sign(pred[moving]))),
        'moving_joint_sign_count': int(moving.sum()),
        'delta_q_cosine_median': float(np.median(cos)),
        'delta_q_cosine_mean': float(np.mean(cos)),
    }


def fit_ridge(xtr, ytr, xte, yte, alpha=10.):
    sx = StandardScaler().fit(xtr)
    sy = np.maximum(ytr.std(0), 1e-5)
    model = Ridge(alpha=alpha).fit(sx.transform(xtr), ytr / sy)
    return metrics(yte, model.predict(sx.transform(xte)) * sy)


def fit_mlp(xtr, ytr, xval, yval, xtest, ytest, label):
    sx = StandardScaler().fit(xtr)
    sy = np.maximum(ytr.std(0), 1e-5).astype(np.float32)
    xtr = sx.transform(xtr).astype(np.float32)
    xval = sx.transform(xval).astype(np.float32)
    xtest = sx.transform(xtest).astype(np.float32)
    ytrn = (ytr / sy).astype(np.float32)
    yvaln = (yval / sy).astype(np.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 192), torch.nn.GELU(),
        torch.nn.Linear(192, 128), torch.nn.GELU(), torch.nn.Linear(128, ytr.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
    device = torch.device('cuda')
    tx = torch.from_numpy(xtr).to(device); ty = torch.from_numpy(ytrn).to(device)
    vx = torch.from_numpy(xval).to(device); vy = torch.from_numpy(yvaln).to(device)
    best, best_state, stale, best_epoch = float('inf'), None, 0, 0
    batch = 2048
    for epoch in range(80):
        net.train()
        order = torch.randperm(len(tx), device=device)
        for ids in order.split(batch):
            loss = (net(tx[ids]) - ty[ids]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vloss = float((net(vx)-vy).square().mean())
        if vloss < best - 1e-5:
            best, stale, best_epoch = vloss, 0, epoch + 1
            best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 10: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): pred = net(torch.from_numpy(xtest).to(device)).cpu().numpy() * sy
    result = metrics(ytest, pred); result['best_epoch'] = best_epoch
    result['best_validation_mse_normalized'] = best
    print(f'{label}: {result}', flush=True)
    return result


print('Sampling matched train and held-out validation clips...', flush=True)
ftr, qtr, ytr, mtr = sample_split('train')
fte, qte, yte, mte = sample_split('val')
unique = np.unique(mtr)
rng.shuffle(unique)
nval = max(1, round(.1 * len(unique)))
probe_val_ids = set(unique[:nval].tolist())
is_probe_val = np.isin(mtr, list(probe_val_ids))
is_fit = ~is_probe_val
print(f'train transitions={len(ytr):,}, probe-val transitions={is_probe_val.sum():,}, '
      f'held-out source-val transitions={len(yte):,}; dims force={ftr.shape[1]}, q={qtr.shape[1]}, dq={ytr.shape[1]}', flush=True)

inputs_tr = {
    'force_only': ftr,
    'state_only_joint_q': qtr,
    'state_q_plus_force': np.concatenate([qtr, ftr], axis=1),
}
inputs_te = {
    'force_only': fte,
    'state_only_joint_q': qte,
    'state_q_plus_force': np.concatenate([qte, fte], axis=1),
}
results = {}
for name in inputs_tr:
    x = inputs_tr[name]
    results[name] = {
        'ridge': fit_ridge(x[is_fit], ytr[is_fit], inputs_te[name], yte),
        'mlp': fit_mlp(x[is_fit], ytr[is_fit], x[is_probe_val], ytr[is_probe_val],
                       inputs_te[name], yte, name),
    }

point_norm = np.linalg.norm(fte.reshape(len(fte), 23, 3), axis=-1)
report = {
    'dataset': 'passive_whole_body_full',
    'prompt_definition': 'F[t,i] = clip_norm(K*(x_ref[t+1,i]-x_ref[t,i]),30 N), world frame',
    'source_manifest': json.loads((CACHE/'train/manifest.json').read_text()),
    'sampling': {'seed': SEED, 'frames_per_clip_max': FRAMES_PER_CLIP,
        'train_clip_count': int(len(np.unique(mtr))), 'train_transition_count': int(len(ytr)),
        'internal_early_stop_clip_count': int(nval),
        'held_out_dataset_split': 'passive_whole_body_full/val',
        'held_out_clip_count': int(len(np.unique(mte))), 'held_out_transition_count': int(len(yte)),
        'grouped_by_clip': True},
    'target': 'joint_pos[t+1]-joint_pos[t], radians; 29 actuated joints',
    'dimensions': {'force': int(ftr.shape[1]), 'force_points': 23, 'state_joint_q': int(qtr.shape[1]), 'delta_q': int(ytr.shape[1])},
    'metrics': results,
    'held_out_force_point_norm_N': {'mean': point_norm.mean(0).tolist(), 'active_fraction': (point_norm>1e-5).mean(0).tolist()},
    'interpretation': 'The cache prompt is generated directly from reference body point displacement, so this tests whether it predicts the corresponding 29-DoF joint increment on unseen clips. It is a data-encoding test, not a policy rollout or causal-force test.'
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
lines = ['# 23-point prompt to joint-motion correspondence', '',
    'Force prompt is built from each of the 23 reference body points moving from frame t to t+1, multiplied by a fixed gain and capped at 30 N per point. Target is the matching 29-joint delta-q. Entire clips in the original validation split are held out.', '',
    f"- Train sample: {len(ytr):,} transitions from {len(np.unique(mtr)):,} clips; held-out: {len(yte):,} transitions from {len(np.unique(mte)):,} clips.",
    '- State baseline uses only current 29-joint pose. State+force tests whether force adds predictive information beyond pose.',
    '- NRMSE is normalized per joint standard deviation (lower better); R², sign agreement, and delta-q cosine are higher better.', '',
    '| Input | Probe | NRMSE ↓ | Macro joint R² ↑ | Moving Δq sign ↑ | Δq cosine median ↑ | MAE (rad) ↓ |',
    '|---|---|---:|---:|---:|---:|---:|']
for inp, models in results.items():
    for model, m in models.items():
        lines.append(f"| {inp} | {model} | {m['nrmse_per_joint_std']:.3f} | {m['r2_macro_joint']:.3f} | {m['moving_joint_sign_accuracy']:.3f} | {m['delta_q_cosine_median']:.3f} | {m['mae_rad']:.5f} |")
lines += ['', '## Interpretation', '', report['interpretation'], '',
    'Because prompt forces are constructed from the very same reference displacement used for the label, this evaluates the encoding and its identifiability across unseen clips. It does not prove that a trained policy can follow the cue.', '']
(OUT/'summary.md').write_text('\n'.join(lines))
print(f'WROTE {OUT}/summary.md', flush=True)
