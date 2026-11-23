"""Compare fixed 8KP force generators using history or single-frame polar features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--force-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--polar", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with np.load(args.input / "data.npz") as d:
        clip, step, split, state = (d[k] for k in ("clip", "step", "split", "state"))
        target = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    if args.force_dir is None:
        root = args.input / "fixed8_jacobian"
        ids, forces = [], []
        for i in range(6):
            with np.load(root / f"shard{i}.npz") as d:
                ids.append(d["ids"])
                forces.append(d["search_fit"])
        ids = np.concatenate(ids)
        order = np.lexsort((ids[:, 1], ids[:, 0]))
        np.testing.assert_array_equal(ids[order], np.c_[clip, step])
        force = np.concatenate(forces)[order]
        result_dir = root / "search_fit"
    else:
        with np.load(args.force_dir / "aligned_forces.npz") as d:
            force = d["force"]
        result_dir = args.force_dir
    output = result_dir / ("polar" if args.polar else "history")
    output.mkdir(exist_ok=True)
    train, test = split == 0, split == 2
    channels = []
    frames = np.arange(len(clip))
    offsets = (0,) if args.polar else (0, 1, 2, 4, 8, 16)
    for offset in offsets:
        previous = (frames - offset).clip(0)
        value = force[previous].copy()
        value[step < offset] = 0.
        if args.polar:
            norm = np.linalg.norm(value, axis=-1, keepdims=True)
            direction = np.divide(value, np.maximum(norm, 1e-8))
            value = np.concatenate((value, direction, np.log1p(norm / .05)), axis=-1)
        channels.append(value.reshape(len(force), -1))
    flat = np.concatenate(channels, axis=1)
    flat = (flat - flat[train].mean(0)) / np.maximum(flat[train].std(0), .1)
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    x = np.c_[state, np.pad(flat, ((0, 0), (0, width - flat.shape[1])))].astype(np.float32)
    pred, _, epoch = fit(x, target, split, args.seed, output / f"model_{args.seed}.pt")
    scores = {k: metrics(target[test, sl], pred[:, sl]) for k, sl in GROUPS.items()}
    (output / f"seed{args.seed}.json").write_text(json.dumps(dict(epoch=epoch, groups=scores), indent=2))
    print(args.seed, {k: v["nrmse"] for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
