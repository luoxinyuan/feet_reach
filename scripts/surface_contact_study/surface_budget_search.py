"""Fixed contact-count surface search on the shared nominal-state benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from active_adaptation.utils.surface_guidance import SurfaceSampler
from scripts.compliance_eval.protocol import load_arrays


METHODS = ("surface_search", "search_points_spring", "random_points_fit", "random_points_spring")


def directed_columns(positions, jacobian, weights):
    displacement = positions[1:] - positions[:-1]
    length = np.linalg.norm(displacement, axis=-1, keepdims=True)
    direction = displacement / np.maximum(length, 1e-8)
    columns = np.einsum("tpcd,tpc->tpd", jacobian[:-1], direction) * weights
    return direction, columns


def amplitudes(columns, target, previous=None):
    """Batched box-constrained ridge fit by projected coordinate descent."""
    gram = columns @ columns.swapaxes(-1, -2)
    rhs = (columns @ target[..., None])[..., 0]
    diagonal = np.diagonal(gram, axis1=-2, axis2=-1) + .15
    if previous is not None:
        diagonal = diagonal + .25
        rhs = rhs + .25 * previous
    value = np.zeros_like(rhs)
    for _ in range(6):
        for k in range(value.shape[-1]):
            other = (gram[..., k, :] * value).sum(-1) - gram[..., k, k] * value[..., k]
            value[..., k] = np.clip((rhs[..., k] - other) / diagonal[..., k], 0., 30.)
    value *= np.minimum(1., 60. / np.maximum(value.sum(-1, keepdims=True), 1e-8))
    return value


def select(columns, target, count):
    """Greedy addition and one exchange sweep with the sampled count fixed."""
    def score(ids):
        cols = columns[:, ids].transpose(1, 0, 2, 3)
        f = amplitudes(cols, target)
        estimate = np.einsum("btk,btkd->btd", f, cols)
        error = (np.square(estimate - target).sum(-1) + .15 * np.square(f).sum(-1))
        return (error / (np.square(target).sum(-1) + .1)).mean(-1)

    pool = np.arange(columns.shape[1])
    chosen = np.empty(0, dtype=int)
    for _ in range(count):
        remaining = np.setdiff1d(pool, chosen)
        candidates = np.c_[np.broadcast_to(chosen, (len(remaining), len(chosen))), remaining]
        chosen = candidates[np.argmin(score(candidates))]
    for slot in range(count):
        remaining = np.setdiff1d(pool, np.delete(chosen, slot))
        candidates = np.broadcast_to(chosen, (len(remaining), count)).copy()
        candidates[:, slot] = remaining
        chosen = candidates[np.argmin(score(candidates))]
    return chosen


def replay(direction, columns, target, positions, fit):
    force = np.zeros_like(direction)
    previous = np.zeros_like(direction[0])
    releasing = False
    for t, (dirs, cols, goal) in enumerate(zip(direction, columns, target)):
        force[t] = previous
        releasing |= (40 - t) * .02 <= np.linalg.norm(previous, axis=-1).max() / 150. + .040001
        if releasing:
            desired = np.zeros_like(previous)
        elif fit:
            desired = dirs * amplitudes(cols, goal, np.maximum((previous * dirs).sum(-1), 0.))[:, None]
        else:
            desired = 80. * (positions[t + 1] - positions[t])
        desired[positions[t, :, 2] <= .06] = 0.
        norms = np.linalg.norm(desired, axis=-1)
        desired *= min(1., 30. / max(norms.max(), 1e-8), 60. / max(norms.sum(), 1e-8))
        change = desired - previous
        previous += min(1., 3. / max(np.linalg.norm(change, axis=-1).max(), 1e-8)) * change
    if len(direction) == 40 and np.linalg.norm(previous) > 1e-6:
        raise RuntimeError("Surface contacts changed before unloading")
    return force


def encode(body, local, force, rotations, nbody):
    order = np.lexsort((local[:, 2], local[:, 1], local[:, 0], body))
    body, local, force = body[order], local[order], (force @ rotations)[:, order]
    visible = np.linalg.norm(force, axis=-1) > 1e-5
    slot = np.cumsum(visible, axis=1) - 1
    t, k = np.nonzero(visible)
    token = np.zeros((len(force), 5, nbody + 7), np.float32)
    s = slot[t, k]
    token[t, s, body[k]] = 1.
    token[t, s, nbody:nbody + 3] = local[k]
    token[t, s, nbody + 3:nbody + 6] = force[t, k]
    token[t, s, -1] = 1.
    return token.reshape(len(force), -1)


def merge(args, study):
    shards = [np.load(args.output / f"shard{i}.npz") for i in range(args.shards)]
    try:
        rows = np.concatenate([shard["rows"] for shard in shards])
        order = np.lexsort((rows[:, 1], rows[:, 0]))
        with np.load(args.input / "data.npz") as data:
            np.testing.assert_array_equal(rows[order], np.c_[data["clip"], data["step"]])
            labels = {key: data[key] for key in
                      ("state", "delta_q", "delta_root_1", "delta_root_5", "split", "clip", "step")}
        features = {key: np.concatenate([shard[key] for shard in shards])[order]
                    for key in METHODS}
        stats = {key: np.concatenate([shard[f"{key}_stats"] for shard in shards])[order]
                 for key in METHODS}
        windows = np.concatenate([shard["windows"] for shard in shards])
    finally:
        for shard in shards:
            shard.close()
    np.savez_compressed(args.output / "data.npz", **labels, **features)
    np.savez_compressed(args.output / "force_stats.npz", windows=windows, **stats)
    manifest = dict(study, upper_body=False, coverage_prior=0., pool=args.pool,
                    methods=list(METHODS), full_span=True, adaptive_release=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    selection = dict(count_probabilities=[.4, .3, .3], contact_window_frames=40,
                     force_target="generalized load of 30 nominal link springs",
                     surface_search="greedy addition and one exchange pass",
                     sense_timing="previous applied force", seeds=[11, 22, 33])
    (args.output / "SELECTION.json").write_text(json.dumps(selection, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--shard", type=int)
    task.add_argument("--merge", action="store_true")
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--pool", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    study = json.loads((args.input / "manifest.json").read_text())
    if args.merge:
        merge(args, study)
        return
    manifest = json.loads((Path(study["source"]) / "manifest.json").read_text())
    cache = json.loads((Path(manifest["force_cache"]) / "manifest.json").read_text())
    datasets = [load_arrays(path)[1] for path in cache["dataset_paths"]]
    geom, names, bodies = cache["geometry"], manifest["joint_names"], study["bodies"]
    sampler = SurfaceSampler(Path("artifacts/surface_compliance/geometry"), bodies, "cpu")
    dense_points = [dict(body=body, offset=[0., 0., 0.]) for body in bodies
                    if not body.endswith("_mimic")]
    weights = np.r_[np.full(3, 1.), np.full(3 + len(names), 1 / .3)]
    features = {method: [] for method in METHODS}
    stats = {method: [] for method in METHODS}
    rows, windows = [], []
    for record in manifest["records"][args.shard::args.shards]:
        clip_id = record["eval_id"]
        rng = np.random.default_rng(712211 + clip_id)
        torch.manual_seed(902211 + clip_id)
        count, start = record["eval_frames"], record["start"]
        frames = slice(start, start + count + 1)
        data = datasets[record["source_index"]]
        q, pos, quat = (np.asarray(data[key][frames], dtype=np.float64)
                        for key in ("joint_pos", "root_pos_w", "root_quat_w"))
        positions, jac = geometry_jacobians(q, pos, quat, names, dense_points, geom)
        dense_force = 80. * np.diff(positions, axis=0)
        goal = np.einsum("tpcd,tpc->td", jac[:-1], dense_force) * weights
        rotations = rot(quat)
        for st in range(0, count, 40):
            en = min(st + 40, count)
            k = int(rng.choice((1, 2, 3), p=(.4, .3, .3)))
            body, triangle, bary = sampler.sample((args.pool,))
            local = (triangle * bary[..., None]).sum(-2).numpy()
            body = body.numpy()
            points = [dict(body=bodies[b], offset=x.tolist()) for b, x in zip(body, local)]
            probes = st + np.minimum([0, 10, 20, 30], en - st - 1)
            paired = np.column_stack((probes, probes + 1)).ravel()
            p, j = geometry_jacobians(q[paired], pos[paired], quat[paired], names, points, geom)
            valid = np.flatnonzero((p[..., 2] > .06).all(axis=0))
            direction = p[1::2, valid] - p[::2, valid]
            direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
            columns = np.einsum("tpcd,tpc->tpd", j[::2, valid], direction) * weights
            optimized = valid[select(columns, goal[probes], k)]
            random = rng.choice(valid, k, replace=False)
            windows.append((clip_id, st, en, k))
            for ids, fit_name, spring_name in (
                (optimized, "surface_search", "search_points_spring"),
                (random, "random_points_fit", "random_points_spring"),
            ):
                f = slice(st, en + 1)
                p, j = geometry_jacobians(q[f], pos[f], quat[f], names, [points[i] for i in ids], geom)
                dirs, cols = directed_columns(p, j, weights)
                for method, solve in ((fit_name, True), (spring_name, False)):
                    force = replay(dirs, cols, goal[st:en], p, solve)
                    token = encode(body[ids], local[ids], force, rotations[st:en], len(bodies))
                    features[method].append(token)
                    norms = np.linalg.norm(force, axis=-1)
                    speed = np.linalg.norm(np.diff(force, axis=0, prepend=np.zeros_like(force[:1])),
                                           axis=-1).max(-1) / .02
                    active = norms > .05
                    opposite = active & ((force * dirs).sum(-1) < 0.)
                    stats[method].append(np.c_[norms.sum(-1), norms.max(-1), active.sum(-1),
                                               speed, opposite.sum(-1)])
            rows.extend((clip_id, t) for t in range(st, en))
    np.savez_compressed(args.output / f"shard{args.shard}.npz",
                        rows=rows, windows=windows,
                        **{k: np.concatenate(v) for k, v in features.items()},
                        **{f"{k}_stats": np.concatenate(v) for k, v in stats.items()})


if __name__ == "__main__":
    main()
