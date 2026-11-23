"""Tune 30-link spring replay using validation-only motion prediction scores."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.segmented_guidance import rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def replay_grid(raw, clip, step, turns, stiffness, cap, rate):
    desired = raw * (stiffness / 80.)
    norm = np.linalg.norm(desired, axis=-1)
    factor = np.minimum(1., np.minimum(30. / np.maximum(norm.max(-1), 1e-8),
                                       cap / np.maximum(norm.sum(-1), 1e-8)))
    desired *= factor[:, None, None]
    _, clip = np.unique(clip, return_inverse=True)
    shape = (clip.max() + 1, step.max() + 1)
    grid = np.zeros((*shape, *desired.shape[1:]), np.float32)
    grid[clip, step] = desired
    rotation_grid = np.broadcast_to(np.eye(3, dtype=np.float32), (*shape, 3, 3)).copy()
    rotation_grid[clip, step] = turns
    observed = np.zeros_like(grid)
    for t in range(shape[1] - 1):
        previous = observed[:, t]
        delta = grid[:, t] - previous
        peak = np.linalg.norm(delta, axis=-1).max(-1)
        alpha = np.minimum(1., .02 * rate / np.maximum(peak, 1e-8))
        observed[:, t + 1] = (previous + alpha[:, None, None] * delta) @ rotation_grid[:, t]
    return observed[clip, step], factor


def load_turns(root, clip):
    study = json.loads((root / "manifest.json").read_text())
    manifest = json.loads((Path(study["source"]) / "manifest.json").read_text())
    cache = json.loads((Path(manifest["force_cache"]) / "manifest.json").read_text())
    datasets = [load_arrays(path)[1] for path in cache["dataset_paths"]]
    turns, ids = [], []
    for record in manifest["records"]:
        start, count = record["start"], record["eval_frames"]
        quat = datasets[record["source_index"]]["root_quat_w"][start:start + count + 1]
        rotation = rot(quat)
        turns.append(rotation[:-1].swapaxes(-1, -2) @ rotation[1:])
        ids.extend([record["eval_id"]] * count)
    np.testing.assert_array_equal(clip, ids)
    return np.concatenate(turns).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--config-list", type=Path)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--noise", type=float, default=0., help="Force observation noise std in N")
    args = parser.parse_args()
    if args.test and args.config_list is None:
        parser.error("--test requires a frozen --config-list")
    torch.set_num_threads(2)
    output = args.input / "link_spring_sweep"
    output.mkdir(exist_ok=True)
    with np.load(args.input / "data.npz") as data:
        clip, step, split = data["clip"], data["step"], data["split"]
        state = data["state"]
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        width = data["greedy3"].shape[1]
    train, valid, test = split == 0, split == 1, split == 2
    with np.load(args.input / "link_spring" / "features.npz") as data:
        raw, baseline = data["raw"], data["applied"]
    turns = load_turns(args.input, clip)
    check, _ = replay_grid(raw, clip, step, turns, 80, 60, 150)
    np.testing.assert_allclose(check, baseline, atol=3e-6)
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    grid = list(itertools.product((20, 40, 80, 160, 320), (20, 40, 60), (50, 150, 300)))
    if args.config_list:
        grid = json.loads(args.config_list.read_text())
    stage = "test" if args.test else "validation"
    for stiffness, cap, rate in grid[args.shard::args.shards]:
        force, factor = replay_grid(raw, clip, step, turns, stiffness, cap, rate)
        norms = np.linalg.norm(force, axis=-1)
        previous = np.r_[0, np.arange(len(clip) - 1)]
        old_in_new_frame = force[previous] @ turns[previous]
        force_rate = np.linalg.norm(force - old_in_new_frame, axis=-1).max(-1) / .02
        force_rate[step == 0] = 0.
        channels = force.reshape(len(state), -1)
        channel_scale = np.maximum(channels[train].std(0), .1)
        channels = (channels - channels[train].mean(0)) / channel_scale
        x = np.c_[state, np.pad(channels, ((0, 0), (0, width - channels.shape[1])))].astype(np.float32)
        config = output / f"k{stiffness}_cap{cap}_rate{rate}"
        config.mkdir(exist_ok=True)
        rows, predictions = [], []
        for seed in args.seeds:
            pred_test, predict, epoch = fit(x, target, split, seed, config / f"decoder_{seed}.pt")
            pred_valid = predict(x[valid])
            row = dict(seed=seed, epoch=epoch, validation={
                name: metrics(target[valid, group], pred_valid[:, group]) for name, group in GROUPS.items()
            })
            if args.test:
                row["test"] = {name: metrics(target[test, group], pred_test[:, group])
                               for name, group in GROUPS.items()}
                predictions.append(pred_test)
            if args.noise:
                sets = [("validation", valid)]
                if args.test:
                    sets.append(("test", test))
                for name, mask in sets:
                    noisy = x[mask].copy()
                    rng = np.random.default_rng(seed + 1000)
                    noisy[:, state.shape[1]:state.shape[1] + channels.shape[1]] += (
                        rng.normal(0., args.noise, (len(noisy), channels.shape[1])) / channel_scale
                    )
                    pred = predict(noisy)
                    row[name + "_noisy"] = {
                        key: metrics(target[mask, group], pred[:, group])
                        for key, group in GROUPS.items()
                    }
            rows.append(row)

        def force_stats(mask):
            total = norms[mask].sum(-1)
            return dict(mean=float(total.mean()), p95=float(np.quantile(total, .95)),
                        maximum=float(total.max()), max_point=float(norms[mask].max()),
                        clipped=float((factor[mask] < .9999).mean()),
                        rate_p95=float(np.quantile(force_rate[mask], .95)),
                        rate_max=float(force_rate[mask].max()))

        stats = {"validation": force_stats(valid)}
        if args.test:
            stats["test"] = force_stats(test)
            np.savez_compressed(config / "test_predictions.npz", prediction=np.stack(predictions),
                                target=target[test], clip=clip[test])
        report = dict(stiffness=stiffness, cap=cap, rate=rate, force=stats, rows=rows,
                      noise_std_N=args.noise)
        (config / f"{stage}_{len(args.seeds)}.json").write_text(json.dumps(report, indent=2))
        avg = np.mean([v["nrmse"] for row in rows for v in row["validation"].values()])
        print(config.name, "validation", round(avg, 5), flush=True)


if __name__ == "__main__":
    main()
