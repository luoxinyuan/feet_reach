"""Replay fixed-layout positional springs on the existing motion split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.body_guidance import limit_force
from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def replay(applied_target):
    observed = np.zeros_like(applied_target)
    for t, desired in enumerate(applied_target[:-1]):
        delta = desired - observed[t]
        peak = np.linalg.norm(delta, axis=-1).max()
        # Four physics steps at 200 Hz with a fixed target and 150 N/s slew.
        observed[t + 1] = observed[t] + min(1., 3. / max(peak, 1e-9)) * delta
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout", type=int, choices=(3, 9, 30), default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("raw", "capped", "applied"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output or args.input / ("link_spring" if args.layout == 30 else f"spring{args.layout}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input / "data.npz") as data:
        split = data["split"]
        state = data["state"]
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        width = data["greedy3"].shape[1]
        clip_ids = data["clip"]
    train, test = split == 0, split == 2
    if args.mode is None:
        study = json.loads((args.input / "manifest.json").read_text())
        source = json.loads((Path(study["source"]) / "manifest.json").read_text())
        cache = json.loads((Path(source["force_cache"]) / "manifest.json").read_text())
        datasets = [load_arrays(path)[1] for path in cache["dataset_paths"]]
        geometry = cache["geometry"]
        if args.layout == 30:
            bodies = ["pelvis"] + [edge["child"] for edge in geometry["edges"]
                                   if not edge["child"].endswith("_mimic")]
            points = [dict(body=body, offset=[0., 0., 0.]) for body in bodies]
        else:
            name = {3: "spring3", 9: "jacobian9"}[args.layout]
            points = source["layouts"][name]
            bodies = [p["body"] for p in points]
        features = {mode: [] for mode in ("raw", "capped", "applied")}
        ids = []
        for record in source["records"]:
            count = record["eval_frames"]
            start = record["start"]
            frames = slice(start, start + count + 1)
            dataset = datasets[record["source_index"]]
            q, pos, quat = (np.asarray(dataset[key][frames], dtype=np.float64)
                            for key in ("joint_pos", "root_pos_w", "root_quat_w"))
            links, _ = geometry_jacobians(q, pos, quat, source["joint_names"], points, geometry)
            raw = (80. * np.diff(links, axis=0)).astype(np.float32)
            capped = limit_force(torch.from_numpy(raw)).numpy()
            rotation = rot(quat[:-1])
            for mode, forces in (("raw", raw), ("capped", capped), ("applied", replay(capped))):
                features[mode].append((forces @ rotation).astype(np.float32))
            ids.extend([record["eval_id"]] * count)
        np.testing.assert_array_equal(ids, clip_ids)
        features = {mode: np.concatenate(values) for mode, values in features.items()}
        np.savez_compressed(output / "features.npz", **features)
        stats = {}
        for mode, forces in features.items():
            norms = np.linalg.norm(forces[test], axis=-1)
            totals = norms.sum(axis=-1)
            stats[mode] = dict(mean_N=float(totals.mean()), p95_N=float(np.quantile(totals, .95)),
                               max_N=float(totals.max()), max_point_N=float(norms.max()),
                               fraction_total_over_60=float((totals > 60.0001).mean()))
        (output / "force_stats.json").write_text(json.dumps(dict(bodies=bodies, test=stats), indent=2))
        return

    with np.load(output / "features.npz") as data:
        channels = data[args.mode].reshape(len(state), -1)
    channels = (channels - channels[train].mean(0)) / np.maximum(channels[train].std(0), .1)
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    x = np.c_[state, np.pad(channels, ((0, 0), (0, width - channels.shape[1])))].astype(np.float32)
    reports = []
    predictions = []
    for seed in (11, 22, 33):
        pred, _, epoch = fit(x, target, split, seed, output / f"{args.mode}_{seed}.pt")
        predictions.append(pred)
        reports.append(dict(seed=seed, best_epoch=epoch, groups={
            key: metrics(target[test, sl], pred[:, sl]) for key, sl in GROUPS.items()
        }))
    (output / f"{args.mode}.json").write_text(json.dumps(reports, indent=2))
    np.savez_compressed(output / f"{args.mode}_predictions.npz",
                        prediction=np.stack(predictions), target=target[test], clip=clip_ids[test])


if __name__ == "__main__":
    main()
