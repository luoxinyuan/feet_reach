"""Compare short-horizon positional springs with fixed or adaptive contacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


LAYOUT = ["pelvis", "left_hand_mimic", "right_hand_mimic", "torso_link"]


def prepare(root, output):
    study = json.loads((root / "manifest.json").read_text())
    source = json.loads((Path(study["source"]) / "manifest.json").read_text())
    cache = json.loads((Path(source["force_cache"]) / "manifest.json").read_text())
    datasets = [load_arrays(path)[1] for path in cache["dataset_paths"]]
    points = [dict(body=b, offset=[0., 0., 0.]) for b in LAYOUT]
    mesh = Path("artifacts/surface_compliance/geometry")
    parts = json.loads((mesh / "manifest.json").read_text())["parts"]
    torso = next(p for p in parts if p["mesh"] == "torso_link")
    tri = np.load(mesh / "triangles.npy", mmap_mode="r")[torso["start"]:torso["end"]]
    centroids = tri.mean(1)
    back = centroids[np.linalg.norm(centroids - [-.10, 0., .17], axis=-1).argmin()]
    points.append(dict(body="torso_link", offset=back.tolist()))
    (output / "points.json").write_text(json.dumps(points, indent=2))
    records = source["records"]
    length = max(r["eval_frames"] for r in records) + 9
    positions = np.zeros((len(records), length, len(points), 3), dtype=np.float32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (len(records), length, 3, 3)).copy()
    ids = []
    for i, record in enumerate(records):
        assert i == record["eval_id"]
        start, size = record["start"], record["eval_frames"]
        n = min(record["frames"], size + 9)
        frames = slice(start, start + n)
        data = datasets[record["source_index"]]
        q, pos, quat = (np.asarray(data[k][frames], dtype=np.float64)
                        for k in ("joint_pos", "root_pos_w", "root_quat_w"))
        links, _ = geometry_jacobians(q, pos, quat, source["joint_names"], points, cache["geometry"])
        positions[i, :n], rotations[i, :n] = links, rot(quat)
        positions[i, n:], rotations[i, n:] = positions[i, n - 1], rotations[i, n - 1]
        ids.extend([i] * size)
    with np.load(root / "data.npz") as d:
        np.testing.assert_array_equal(ids, d["clip"])
    np.savez_compressed(output / "geometry.npz", positions=positions, rotations=rotations)


def contact_schedule(mode, position, rotation, desired, end, horizon):
    """Choose at each window start from the current short lookahead only."""
    clips, _, points = position.shape[:3]
    enabled = np.zeros((clips, end, points), dtype=bool)
    if mode in ("fixed", "fixed_torso", "hands", "back") or mode.startswith("mixed"):
        trunk = 4 if "back" in mode else (3 if mode == "fixed_torso" else 0)
        ids = [1, 2] if mode == "hands" else [trunk, 1, 2]
        enabled[:, :, ids] = True
        return enabled, len(ids)
    if mode.startswith("prune"):
        norm = np.linalg.norm(desired[:, :, :3], axis=-1)
        threshold = int(mode.removeprefix("prune")) / 100.
        active = np.zeros((clips, 3), dtype=bool)
        for t in range(end):
            ratio = norm[:, t] / np.maximum(norm[:, t].max(-1, keepdims=True), 1e-8)
            active = (ratio >= np.where(active, threshold * .7, threshold)) & (norm[:, t] > .02)
            enabled[:, t, :3] = active
        return enabled, 3
    maximum = 2 if "two" in mode else 3
    period = 15 if "short" in mode else 30
    ratio = .30 if "sparse" in mode else .15
    local = (position - position[:, :, :1]) @ rotation
    for start in range(0, end, period):
        finish = min(start + period, end)
        # Root motion and independent arm motion identify distinct guidance needs.
        future = start + horizon
        body_motion = np.linalg.norm(position[:, future, 0] - position[:, start, 0], axis=-1)
        torso_motion = np.linalg.norm(position[:, future, 3] - position[:, start, 3], axis=-1)
        independent = np.linalg.norm(local[:, future, 1:3] - local[:, start, 1:3], axis=-1)
        world = np.linalg.norm(position[:, future, 1:3] - position[:, start, 1:3], axis=-1)
        spin = np.linalg.norm(rotation[:, future] - rotation[:, start], axis=(1, 2)) * .18
        arms = np.minimum(world, independent + .55 * spin[:, None])
        root = np.maximum(body_motion, torso_motion)
        if mode == "three_pelvis":
            root, torso_motion = body_motion, body_motion
        scores = np.c_[root, arms]
        indices = np.argsort(-scores, axis=-1)
        rows = np.arange(clips)
        for rank in range(maximum):
            point = indices[:, rank].copy()
            active = scores[rows, point] >= ratio * scores[rows, indices[:, 0]]
            active &= scores[rows, point] > 1e-6
            point[(point == 0) & (torso_motion > 1.15 * body_motion)] = 3
            enabled[rows, start:finish, point] |= active[:, None]
    return enabled, maximum


def replay(position, rotation, clip, step, mode, horizon, gain=80.):
    end = step.max() + 1
    desired = (gain / horizon) * (position[:, horizon:end + horizon] - position[:, :end])
    if mode.startswith("mixed"):
        group = [1, 2] if "arms" in mode else [4 if "back" in mode else 0]
        short = int(mode[-1])
        desired[:, :, group] = (gain / short) * (
            position[:, short:end + short, group] - position[:, :end, group]
        )
    enabled, maximum = contact_schedule(mode, position, rotation, desired, end, horizon)
    norm = np.linalg.norm(desired, axis=-1)
    requested = norm * enabled
    factor = np.minimum(1., np.minimum(30. / np.maximum(requested.max(-1), 1e-8),
                                       20. / np.maximum(requested.sum(-1), 1e-8)))
    desired *= factor[..., None, None]
    observed = np.zeros_like(desired)
    active = enabled[:, 0].copy()
    for t in range(end - 1):
        force = observed[:, t]
        active &= enabled[:, t] | (np.linalg.norm(force, axis=-1) > 1e-6)
        pending = enabled[:, t] & ~active
        # Drain the old contact before activating its replacement.
        safe = ~(active & ~enabled[:, t]).any(-1)
        active |= pending & safe[:, None]
        goal = desired[:, t] * (active & enabled[:, t])[..., None]
        delta = goal - force
        alpha = np.minimum(1., 3. / np.maximum(np.linalg.norm(delta, axis=-1).max(-1), 1e-8))
        observed[:, t + 1] = force + alpha[..., None, None] * delta
    world_force = observed[clip, step]
    force = world_force @ rotation[clip, step]
    observed_count = (np.linalg.norm(world_force, axis=-1) > 1e-5).sum(-1)
    assert observed_count.max() <= maximum
    return force, world_force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test and args.configs is None:
        parser.error("--test requires a frozen configuration list")
    torch.set_num_threads(2)
    root = args.input
    output = root / "sparse_lookahead_spring"
    output.mkdir(exist_ok=True)
    if args.prepare:
        prepare(root, output)
        return
    with np.load(root / "data.npz") as d:
        clip, step, split, state = d["clip"], d["step"], d["split"], d["state"]
        target = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    with np.load(output / "geometry.npz") as d:
        position, rotation = d["positions"], d["rotations"]
    train, valid, test = split == 0, split == 1, split == 2
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    grid = [[mode, horizon, 80] for mode in
            ("fixed", "fixed_torso", "two_sparse", "two_short", "three_pelvis", "three_short")
            for horizon in (1, 2, 3, 5, 8)]
    if args.configs:
        grid = json.loads(args.configs.read_text())
    for mode, horizon, gain in grid[args.shard::args.shards]:
        forces, world = replay(position, rotation, clip, step, mode, horizon, gain)
        channels = forces.reshape(len(state), -1)
        channels = (channels - channels[train].mean(0)) / np.maximum(channels[train].std(0), .1)
        x = np.c_[state, np.pad(channels, ((0, 0), (0, width - channels.shape[1])))].astype(np.float32)
        path = output / f"{mode}_h{horizon}_k{gain}"
        path.mkdir(exist_ok=True)
        stage = "test" if args.test else "validation"
        mask = test if args.test else valid
        rows, predictions = [], []
        for seed in args.seeds:
            pred_test, predict, epoch = fit(x, target, split, seed, path / f"{stage}_{seed}.pt")
            pred = pred_test if args.test else predict(x[valid])
            predictions.append(pred)
            rows.append(dict(seed=seed, epoch=epoch, groups={
                name: metrics(target[mask, sl], pred[:, sl]) for name, sl in GROUPS.items()
            }))
        if args.test:
            np.savez_compressed(path / "predictions.npz", prediction=np.stack(predictions),
                                target=target[test], clip=clip[test])
        norms = np.linalg.norm(world[mask], axis=-1)
        total = norms.sum(-1)
        change = np.linalg.norm(world[1:] - world[:-1], axis=-1).max(-1) / .02
        rate = change[mask[1:] & (step[1:] > 0)]
        assert rate.max() <= 150.01 and total.max() <= 20.001
        counts = np.bincount((norms > .05).sum(-1), minlength=5)
        report = dict(mode=mode, horizon=horizon, gain=gain, rows=rows,
                      force=dict(mean=float(total.mean()), p95=float(np.quantile(total, .95)),
                                 max=float(total.max()), rate_p95=float(np.quantile(rate, .95)),
                                 counts=counts.tolist()))
        (path / f"{stage}_{len(args.seeds)}.json").write_text(json.dumps(report, indent=2))
        nrmse = np.mean([m["nrmse"] for r in rows for m in r["groups"].values()])
        print(mode, horizon, gain, round(nrmse, 5), flush=True)


if __name__ == "__main__":
    main()
