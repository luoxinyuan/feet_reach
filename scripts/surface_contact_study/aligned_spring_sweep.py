"""Measure the contact-count tradeoff with exactly aligned positional springs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics
from scripts.surface_contact_study.fixed8_learned_force import input_path


LAYOUTS = {
    "hands_back": (0, 1, 2, 3),
    "hands_shoulders": (1, 2, 4, 5),
    "hands_elbows": (1, 2, 6, 7),
    "pelvis_hands_shoulders": (0, 1, 2, 4, 5),
    "pelvis_hands_elbows": (0, 1, 2, 6, 7),
    "back_hands_elbows": (1, 2, 3, 6, 7),
    "pelvis_hands_back_shoulders": (0, 1, 2, 3, 4, 5),
    "pelvis_hands_back_elbows": (0, 1, 2, 3, 6, 7),
    "pelvis_hands_elbows_shoulders": (0, 1, 2, 4, 5, 6, 7),
    "all": tuple(range(8)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--configs", type=Path)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    if args.test and args.configs is None:
        p.error("--test requires a frozen configuration list")
    torch.set_num_threads(2)
    root = args.input
    out = root / "aligned_spring_sweep"
    out.mkdir(exist_ok=True)
    with np.load(root / "data.npz") as d:
        split, state, step = d["split"], d["state"], d["step"]
        target = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    train, val, test = split == 0, split == 1, split == 2
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    configs = [[name, h] for name in LAYOUTS for h in (2, 3, 5)]
    if args.configs:
        configs = json.loads(args.configs.read_text())
    for config in configs[args.shard::args.shards]:
        name, horizon, *settings = config
        gain, trunk_horizon = settings or (40, horizon)
        selected = LAYOUTS[name]
        with np.load(input_path(root / "fixed8_learned_force", horizon)) as d:
            force = d["displacement"][:, selected] * (gain / horizon)
        if trunk_horizon != horizon:
            trunks = [i for i, slot in enumerate(selected) if slot in (0, 3)]
            with np.load(input_path(root / "fixed8_learned_force", trunk_horizon)) as d:
                force[:, trunks] = d["displacement"][:, [selected[i] for i in trunks]] * (gain / trunk_horizon)
        force[step == 0] = 0.
        norm = np.linalg.norm(force, axis=-1)
        force *= np.minimum(1., 1.5 / np.maximum(norm, 1e-8))[..., None]
        norm = np.linalg.norm(force, axis=-1)
        flat = force.reshape(len(force), -1)
        flat = (flat - flat[train].mean(0)) / np.maximum(flat[train].std(0), .1)
        x = np.c_[state, np.pad(flat, ((0, 0), (0, width - flat.shape[1])))].astype(np.float32)
        suffix = f"_gain{gain}_trunk{trunk_horizon}" if settings else ""
        path = out / f"{name}_h{horizon}{suffix}"
        path.mkdir(exist_ok=True)
        subset = test if args.test else val
        rows = []
        for seed in ((11, 22, 33) if args.test else (11,)):
            held, predict, _ = fit(x, target, split, seed, path / f"probe_{seed}.pt")
            pred = held if args.test else predict(x[val])
            rows.append({k: metrics(target[subset, sl], pred[:, sl]) for k, sl in GROUPS.items()})
        result = dict(layout=name, horizon=horizon, max_contacts=len(LAYOUTS[name]), rows=rows,
                      gain=gain, trunk_horizon=trunk_horizon,
                      mean_contacts=float((norm[subset] > .05).sum(-1).mean()),
                      mean_total_N=float(norm[subset].sum(-1).mean()))
        stage = "test" if args.test else "validation"
        (path / f"{stage}.json").write_text(json.dumps(result, indent=2))
        print(name, horizon, np.mean([v["nrmse"] for r in rows for v in r.values()]), flush=True)


if __name__ == "__main__":
    main()
