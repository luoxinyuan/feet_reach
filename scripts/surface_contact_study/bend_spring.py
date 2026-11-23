"""Replay a sparse spring with a torso surface contact dedicated to tilt."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def result_dir(base, gain, gate, crouch_guard, relative=False, planar=False):
    family = "relative_spring" if relative else "bend_spring"
    suffix = "_guard" if crouch_guard else ("_planar" if planar else "")
    return base / family / (f"k{gain}_g{gate}" + suffix)


def geometry(q, root, quat, names, points, model, relative=False):
    offsets = [dict(body="torso_link", offset=axis.tolist()) for axis in np.eye(3)]
    positions, jac = geometry_jacobians(q, root, quat, names, points + offsets, model)
    torso = (positions[:, len(points):] - positions[:, 3:4]).transpose(0, 2, 1)
    common = (positions[:, :len(points)], rot(quat), torso)
    if relative:
        return (*common, None, None, None)
    tilt = [names.index(f"waist_{axis}_joint") for axis in ("roll", "pitch")]
    knees = [names.index(f"{side}_knee_joint") for side in ("left", "right")]
    return (*common, q[:, tilt], jac[:, 4][..., np.asarray(tilt) + 6], q[:, knees])


def spring(positions, root_rot, torso_rot, waist, back_jac, knees, horizon, gain, gate,
           crouch_guard=False, relative=False, planar=False):
    delta = positions[horizon:] - positions[:-horizon]
    desired = delta * (40. / horizon)
    if relative:
        back_local = (positions[:, 4:5] - positions[:, :1]) @ root_rot
        relative_delta = back_local[horizon:] - back_local[:-horizon]
        if planar:
            relative_delta[..., 2] = 0.
        back = (relative_delta @ root_rot[:-horizon].transpose(0, 2, 1))[:, 0]
    else:
        local_tilt = (back_jac[:-horizon] @ (waist[horizon:] - waist[:-horizon])[..., None])[..., 0]
        local_tilt = (local_tilt[:, None, :] @ root_rot[:-horizon])[:, 0]
        local_tilt[:, 2] = 0.
        back = (local_tilt[:, None, :] @ root_rot[:-horizon].transpose(0, 2, 1))[:, 0]
    back_size = np.linalg.norm(back, axis=-1)
    pelvis_size = np.linalg.norm(delta[:, 0], axis=-1)
    if crouch_guard:
        knee_delta = knees[horizon:] - knees[:-horizon]
        dz = delta[:, 0, 2]
        crouch = ((np.abs(dz) > .001) & (dz * knee_delta[:, 0] < 0.)
                  & (dz * knee_delta[:, 1] < 0.) & (np.abs(knee_delta).min(-1) > .003))

    # Subtract the exact motion of a point rigidly attached to the torso.
    from_torso = positions[:-horizon, 1:3] - positions[:-horizon, 3:4]
    follow = (from_torso @ torso_rot[:-horizon] @ torso_rot[horizon:].transpose(0, 2, 1)
              + positions[horizon:, 3:4])
    independent = positions[horizon:, 1:3] - follow

    previous = np.zeros_like(desired[0])
    forces = np.zeros_like(desired)
    active = np.zeros(desired.shape[1], bool)
    back_on = False
    for t in range(len(desired)):
        forces[t] = previous
        # Hysteresis on both absolute signal and relative dominance of waist motion.
        factor = .7 if back_on else 1.
        back_on = back_size[t] > factor * max(.0004, gate * pelvis_size[t])
        back_on &= not (crouch_guard and crouch[t])
        goal = desired[t].copy()
        goal[3:] = 0.
        if back_on:
            goal[0] = 0.
            goal[1:3] = independent[t] * (40. / horizon)
            goal[4] = gain * back[t]
        ordinary = np.linalg.norm(goal[:3], axis=-1)
        threshold = np.maximum(.02, ordinary.max() * np.where(active[:3], .07, .10))
        goal[:3] *= (ordinary >= threshold)[:, None]
        requested = np.linalg.norm(goal, axis=-1)
        goal *= min(1., 20. / max(requested.sum(), 1e-8), 30. / max(requested.max(), 1e-8))
        # Release the pelvis or the back before enabling its replacement.
        changing = (back_on and np.linalg.norm(previous[0]) > 1e-7
                    or not back_on and np.linalg.norm(previous[4]) > 1e-7)
        if changing:
            goal[[0, 4]] = 0.
        active = np.linalg.norm(goal, axis=-1) > 1e-7
        delta_f = goal - previous
        previous += min(1., 3. / max(np.linalg.norm(delta_f, axis=-1).max(), 1e-8)) * delta_f
    return forces


def make_features(q, pos, quat, names, points, model, steps, gain, gate, crouch_guard,
                  relative, planar):
    g = geometry(q, pos, quat, names, points, model, relative=relative)
    padded = [np.concatenate((v, np.repeat(v[-1:], 5, axis=0))) if v is not None else None
              for v in g]
    world = spring(*padded, horizon=5, gain=gain, gate=gate, crouch_guard=crouch_guard,
                   relative=relative, planar=planar)[:steps]
    return world @ g[1][:steps], world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gain", type=int, required=True)
    parser.add_argument("--gate", type=float, default=.6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--crouch-guard", action="store_true")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--planar", action="store_true")
    args = parser.parse_args()
    if args.relative and args.crouch_guard:
        parser.error("relative springs do not use the crouch guard")
    if args.planar and not args.relative:
        parser.error("--planar requires --relative")
    torch.set_num_threads(2)
    base = args.input
    out = result_dir(base, args.gain, args.gate, args.crouch_guard, args.relative, args.planar)
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads((base / "manifest.json").read_text())
    src = json.loads((Path(study["source"]) / "manifest.json").read_text())
    cache = json.loads((Path(src["force_cache"]) / "manifest.json").read_text())
    datasets = [load_arrays(path)[1] for path in cache["dataset_paths"]]
    points = json.loads((base / "sparse_lookahead_spring" / "points.json").read_text())
    with np.load(base / "data.npz") as d:
        state, split, clip, step = d["state"], d["split"], d["clip"], d["step"]
        y = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    forces, worlds, rows = [], [], []
    for record in src["records"]:
        t, n = record["start"], record["eval_frames"]
        frames = slice(t, t + min(n + 5, record["frames"]))
        data = datasets[record["source_index"]]
        q, p, r = (np.asarray(data[name][frames], dtype=np.float64)
                   for name in ("joint_pos", "root_pos_w", "root_quat_w"))
        f, world = make_features(q, p, r, src["joint_names"], points, cache["geometry"],
                                 n, args.gain, args.gate, args.crouch_guard, args.relative, args.planar)
        forces.append(f.astype(np.float32))
        worlds.append(world.astype(np.float32))
        rows.extend([record["eval_id"]] * n)
    np.testing.assert_array_equal(rows, clip)
    force, world = np.concatenate(forces), np.concatenate(worlds)
    norms = np.linalg.norm(world, axis=-1)
    rates = np.linalg.norm(np.diff(world, axis=0), axis=-1).max(-1)[step[1:] > 0] / .02
    assert (norms > 1e-5).sum(-1).max() <= 3
    assert not ((norms[:, 0] > 1e-5) & (norms[:, 4] > 1e-5)).any()
    assert norms.sum(-1).max() <= 20.001 and rates.max() <= 150.01
    train, valid, test = split == 0, split == 1, split == 2
    flat = force.reshape(len(force), -1)
    sm, ss = state[train].mean(0), np.maximum(state[train].std(0), .01)
    fm, fs = flat[train].mean(0), np.maximum(flat[train].std(0), .1)
    x = np.c_[(state - sm) / ss, np.pad((flat - fm) / fs, ((0, 0), (0, width - flat.shape[1])))]
    mask = test if args.test else valid
    norm = norms[mask]
    stats = dict(mean_total_N=float(norm.sum(-1).mean()),
                 back_active_fraction=float((norm[:, 4] > .05).mean()),
                 count=np.bincount((norm > .05).sum(-1), minlength=4).tolist())
    reports = []
    for seed in args.seeds:
        pred_test, predict, epoch = fit(x.astype(np.float32), y, split, seed, out / f"model_{seed}.pt")
        pred = pred_test if args.test else predict(x[valid].astype(np.float32))
        reports.append(dict(seed=seed, epoch=epoch, groups={
            key: metrics(y[mask, sl], pred[:, sl]) for key, sl in GROUPS.items()
        }))
    np.savez_compressed(out / "normalization.npz", state_mean=sm, state_scale=ss,
                        force_mean=fm, force_scale=fs)
    result = dict(gain=args.gain, gate=args.gate, crouch_guard=args.crouch_guard,
                  relative=args.relative, planar=args.planar,
                  stats=stats, rows=reports)
    (out / ("test.json" if args.test else "validation.json")).write_text(json.dumps(result, indent=2))
    print(args.gain, args.gate, stats, flush=True)


if __name__ == "__main__":
    main()
