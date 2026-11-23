"""Sample a contact count, then exhaustively search fixed 8KP subsets."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics
from scripts.surface_contact_study.masked_spring import load_study


METHODS = ("search_fit", "search_spring", "random_spring", "random_fit")
SUBSETS = {k: np.array(list(combinations(range(8), k))) for k in (1, 2, 3)}


def limit(force):
    norms = np.linalg.norm(force, axis=-1)
    scale = np.minimum(1., 20. / np.maximum(norms.sum(-1), 1e-8))
    return force * scale[..., None, None]


def solve(matrix, target, spring, previous=None):
    """Regularized Jacobian fit, then project away opposing force."""
    transpose = matrix.swapaxes(-1, -2)
    ridge = .60 if previous is None else .90
    h = transpose @ matrix + ridge * np.eye(matrix.shape[-1])
    rhs = (transpose @ target[..., None])[..., 0] + .60 * spring.reshape(h.shape[:-1])
    if previous is not None:
        rhs += .30 * previous.ravel()
    f = np.linalg.solve(h, rhs[..., None])[..., 0].reshape(spring.shape)
    square = np.square(spring).sum(-1, keepdims=True)
    dot = (f * spring).sum(-1, keepdims=True)
    f -= np.minimum(dot, 0.) / np.maximum(square, 1e-12) * spring
    f = np.where(square > 1e-10, f, 0.)
    return limit(f)


def design(j, ids, weight):
    # T x subsets x generalized coordinates x stacked point force XYZ.
    columns = j[:, ids].transpose(0, 1, 4, 2, 3)
    return columns.reshape(*columns.shape[:3], -1) * weight[None, None, :, None]


def pick(j, target, spring, count, weight):
    subsets = SUBSETS[count]
    matrices = design(j, subsets, weight)
    priors = spring[:, subsets]
    f = solve(matrices, target[:, None], priors)
    delta = (matrices @ f.reshape(*f.shape[:2], -1, 1))[..., 0] - target[:, None]
    cost = (np.square(delta).sum(-1) + .60 * np.square(f - priors).sum((-1, -2)))
    score = (cost / (np.square(target).sum(-1, keepdims=True) + .01)).mean(0)
    return subsets[score.argmin()]


def rollout(j, target, prior, ids, weight):
    matrix = design(j, ids[None], weight)[:, 0]
    prior = prior[:, ids]
    output = []
    for fit_force in (True, False):
        observed = np.zeros((len(j), 8, 3))
        previous = np.zeros((len(ids), 3))
        releasing = False
        for t in range(len(j)):
            observed[t, ids] = previous
            peak = np.linalg.norm(previous, axis=-1).max()
            releasing |= (len(j) - t) * .02 <= peak / 150. + .040001
            if releasing:
                requested = np.zeros_like(previous)
            elif fit_force:
                requested = solve(matrix[t], target[t], prior[t], previous)
            else:
                requested = limit(prior[t])
            change = requested - previous
            previous += min(1., 3. / max(np.linalg.norm(change, axis=-1).max(), 1e-8)) * change
        assert np.linalg.norm(previous) < 1e-6
        output.append(observed)
    return output


def prepare(root, out, shard, shards):
    manifest, geom, points, paths = load_study(root)
    datasets = [load_arrays(path)[1] for path in paths]
    names = manifest["joint_names"]
    dense = [dict(body="pelvis", offset=[0.] * 3)]
    dense.extend(dict(body=e["child"], offset=[0.] * 3) for e in geom["edges"]
                 if not e["child"].endswith("_mimic"))
    weight = np.r_[np.full(3, .7), np.full(3, .7 / .3), np.full(len(names), 1. / .3)]
    results = {name: [] for name in METHODS}
    ids, windows = [], []
    for rec in manifest["records"][shard::shards]:
        clip, start, count = rec["eval_id"], rec["start"], rec["eval_frames"]
        rng = np.random.default_rng(935171 + clip)
        data = datasets[rec["source_index"]]
        sl = slice(start, start + min(count + 5, rec["frames"]))
        q, p, r = (np.asarray(data[k][sl], dtype=np.float64)
                   for k in ("joint_pos", "root_pos_w", "root_quat_w"))
        pos, j = geometry_jacobians(q, p, r, names, points + dense, geom)
        delta = pos[np.minimum(np.arange(count) + 5, len(pos) - 1)] - pos[:count]
        spring = 8. * delta
        goal = np.einsum("tpcd,tpc->td", j[:count, 8:], limit(spring[:, 8:])) * weight
        j, spring = j[:count, :8], spring[:, :8]
        rotations = rot(r[:count])
        for st in range(0, count, 40):
            en = min(count, st + 40)
            k = int(rng.choice((1, 2, 3), p=(.4, .3, .3)))
            probe = np.minimum(st + np.array([0, 10, 20, 30]), en - 1)
            best = pick(j[probe], goal[probe], spring[probe], k, weight)
            random = rng.choice(8, k, replace=False)
            windows.append((clip, st, en, k, *np.pad(best, (0, 3-k), constant_values=-1)))
            f, s = rollout(j[st:en], goal[st:en], spring[st:en], best, weight)
            rf, rs = rollout(j[st:en], goal[st:en], spring[st:en], random, weight)
            for name, w in zip(METHODS, (f, s, rs, rf)):
                results[name].append((w @ rotations[st:en]).astype(np.float32))
            ids.extend((clip, t) for t in range(st, en))
    np.savez_compressed(out / f"shard{shard}.npz", ids=ids, windows=windows,
                        **{name: np.concatenate(parts) for name, parts in results.items()})


def evaluate(root, out, name, shards):
    data = [np.load(out / f"shard{i}.npz") for i in range(shards)]
    try:
        ids = np.concatenate([d["ids"] for d in data])
        order = np.lexsort((ids[:, 1], ids[:, 0]))
        f = np.concatenate([d[name] for d in data])[order]
        windows = np.concatenate([d["windows"] for d in data])
    finally:
        for d in data:
            d.close()
    with np.load(root / "data.npz") as d:
        clip, step, split, state = (d[k] for k in ("clip", "step", "split", "state"))
        np.testing.assert_array_equal(ids[order], np.c_[clip, step])
        y = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    train, test = split == 0, split == 2
    norms = np.linalg.norm(f[test], axis=-1)
    active = (norms > .05).sum(-1)
    assert active.max() <= 3 and norms.sum(-1).max() <= 20.001
    channels = f.reshape(len(f), -1)
    sm, ss = state[train].mean(0), np.maximum(state[train].std(0), .01)
    fm, fs = channels[train].mean(0), np.maximum(channels[train].std(0), .1)
    x = np.c_[(state - sm) / ss,
              np.pad((channels - fm) / fs, ((0, 0), (0, width - channels.shape[1])))].astype(np.float32)
    path = out / name
    path.mkdir(exist_ok=True)
    rows, predictions = [], []
    for seed in (11, 22, 33):
        pred, _, epoch = fit(x, y, split, seed, path / f"model_{seed}.pt")
        predictions.append(pred)
        rows.append(dict(seed=seed, epoch=epoch, groups={
            key: metrics(y[test, sl], pred[:, sl]) for key, sl in GROUPS.items()
        }))
    held = windows[np.isin(windows[:, 0], np.unique(clip[test]))]
    stats = dict(mean_force=float(norms.sum(-1).mean()),
                 p95_force=float(np.quantile(norms.sum(-1), .95)),
                 active_counts=np.bincount(active, minlength=4).tolist(),
                 requested_counts=np.bincount(held[:, 3], minlength=4).tolist(),
                 searched_point_counts=np.bincount(held[:, 4:][held[:, 4:] >= 0], minlength=8).tolist())
    np.savez_compressed(path / "predictions.npz", prediction=np.stack(predictions),
                        target=y[test], clip=clip[test])
    (path / "result.json").write_text(json.dumps(dict(rows=rows, stats=stats), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--eval", choices=METHODS)
    args = parser.parse_args()
    torch.set_num_threads(2 if args.eval else 1)
    out = args.input / "fixed8_jacobian"
    out.mkdir(exist_ok=True)
    if args.eval:
        evaluate(args.input, out, args.eval, args.shards)
    else:
        prepare(args.input, out, args.shard, args.shards)


if __name__ == "__main__":
    main()
