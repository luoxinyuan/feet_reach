"""Compare fixed spring layouts with a blockwise random contact mask."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def point_layout(root):
    points = json.loads((root / "sparse_lookahead_spring" / "points.json").read_text())
    return [points[i] for i in (0, 1, 2, 4)] + [
        dict(body=f"{side}_{joint}_link", offset=[0., 0., 0.])
        for joint in ("shoulder_roll", "elbow") for side in ("left", "right")
    ]


def replay(positions, rotations, steps, count, masked, seed):
    points = positions[:, :count]
    t = np.arange(steps)
    delta = points[np.minimum(t + 5, len(points) - 1)] - points[:steps]
    requested = 8. * delta
    norms = np.linalg.norm(requested, axis=-1)
    # Cap the dense prompt before masking so retained point forces keep their scale.
    scale = np.minimum(1., 20. / np.maximum(norms.sum(-1), 1e-8))
    requested *= scale[:, None, None]
    force = np.zeros_like(requested)
    previous = np.zeros_like(requested[0])
    rng = np.random.default_rng(seed)
    windows = []
    for start in range(0, steps, 40):
        end = min(start + 40, steps)
        cap = int(rng.choice((1, 2, 3), p=(.5, .4, .1)))
        ranks = rng.random(8)
        keep = np.argsort(ranks[:count])[:cap] if masked else np.arange(count)
        windows.append((start, end, len(keep)))
        releasing = False
        for i in range(start, end):
            force[i] = previous
            target = requested[i].copy()
            if masked:
                goal = np.zeros_like(target)
                goal[keep] = target[keep]
                releasing |= (end - i) * .02 <= np.linalg.norm(previous, axis=-1).max() / 150. + .040001
                target = np.zeros_like(goal) if releasing else goal
            change = target - previous
            previous += min(1., 3. / max(np.linalg.norm(change, axis=-1).max(), 1e-8)) * change
        if masked and end < steps:
            assert np.linalg.norm(previous) < 1e-6
    return force @ rotations[:steps], force, windows


def load_study(root):
    study = json.loads((root / "manifest.json").read_text())
    manifest = json.loads((Path(study["source"]) / "manifest.json").read_text())
    cache = json.loads((Path(manifest["force_cache"]) / "manifest.json").read_text())
    return manifest, cache["geometry"], point_layout(root), cache["dataset_paths"]


def prepare(root, out):
    manifest, geom, points, paths = load_study(root)
    datasets = [load_arrays(path)[1] for path in paths]
    arrays, ids = {}, []
    for r in manifest["records"]:
        clip_id, steps, start = r["eval_id"], r["eval_frames"], r["start"]
        data = datasets[r["source_index"]]
        frames = slice(start, start + min(steps + 5, r["frames"]))
        q, pos, quat = (np.asarray(data[k][frames], dtype=np.float64)
                        for k in ("joint_pos", "root_pos_w", "root_quat_w"))
        p, _ = geometry_jacobians(q, pos, quat, manifest["joint_names"], points, geom)
        arrays[f"p{clip_id}"] = p.astype(np.float32)
        arrays[f"r{clip_id}"] = rot(quat).astype(np.float32)
        ids.extend([clip_id] * steps)
    with np.load(root / "data.npz") as data:
        np.testing.assert_array_equal(ids, data["clip"])
    np.savez_compressed(out / "geometry.npz", **arrays)
    (out / "points.json").write_text(json.dumps(points, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout", choices=("mask4", "mask8", "dense4", "dense8"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.input
    out = root / "masked_spring"
    out.mkdir(exist_ok=True)
    if args.layout is None:
        prepare(root, out)
        return
    with np.load(root / "data.npz") as data:
        clip, step, split, state = (data[k] for k in ("clip", "step", "split", "state"))
        y = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        width = data["greedy3"].shape[1]
    train, test = split == 0, split == 2
    count = int(args.layout[-1])
    masked = args.layout.startswith("mask")
    forces, worlds, windows = [], [], []
    with np.load(out / "geometry.npz") as data:
        for c, steps in zip(*np.unique(clip, return_counts=True)):
            f, w, blocks = replay(data[f"p{c}"], data[f"r{c}"], steps, count, masked, 882211 + c)
            forces.append(f)
            worlds.append(w)
            windows.extend((c, a, b, k) for a, b, k in blocks)
    force, world = np.concatenate(forces), np.concatenate(worlds)
    norms = np.linalg.norm(world, axis=-1)
    total = norms.sum(-1)
    rate = np.linalg.norm(np.diff(world, axis=0), axis=-1).max(-1) / .02
    assert total.max() <= 20.001 and rate[step[1:] > 0].max() <= 150.01
    if masked:
        assert (norms > 1e-5).sum(-1).max() <= 3
    flat = force.reshape(len(force), -1)
    sm, ss = state[train].mean(0), np.maximum(state[train].std(0), .01)
    fm, fs = flat[train].mean(0), np.maximum(flat[train].std(0), .1)
    x = np.c_[(state - sm) / ss,
              np.pad((flat - fm) / fs, ((0, 0), (0, width - flat.shape[1])))].astype(np.float32)
    path = out / args.layout
    path.mkdir(exist_ok=True)
    rows, predictions = [], []
    for seed in (11, 22, 33):
        pred, _, epoch = fit(x, y, split, seed, path / f"model_{seed}.pt")
        predictions.append(pred)
        rows.append(dict(seed=seed, epoch=epoch, groups={
            name: metrics(y[test, sl], pred[:, sl]) for name, sl in GROUPS.items()
        }))
    blocks = np.array(windows)
    test_blocks = blocks[np.isin(blocks[:, 0], np.unique(clip[test]))]
    stats = dict(mean_total_N=float(total[test].mean()), p95_total_N=float(np.quantile(total[test], .95)),
                 active_counts=np.bincount((norms[test] > .05).sum(-1), minlength=9).tolist(),
                 requested_counts=np.bincount(test_blocks[:, 3], minlength=9).tolist())
    np.savez_compressed(path / "normalization.npz", sm=sm, ss=ss, fm=fm, fs=fs)
    np.savez_compressed(path / "predictions.npz", prediction=np.stack(predictions), target=y[test],
                        clip=clip[test])
    (path / "result.json").write_text(json.dumps(dict(rows=rows, stats=stats), indent=2))


if __name__ == "__main__":
    main()
