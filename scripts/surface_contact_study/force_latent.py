"""Learn a force-only bottleneck by supervising the motion prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


class ForceLatentDecoder(nn.Module):
    def __init__(self, state_dim, force_dim, latent_dim, target_dim):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(nn.Linear(force_dim, 128), nn.SiLU(),
                                     nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(state_dim + latent_dim, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, target_dim))

    def forward(self, features):
        state = features[:, :self.state_dim]
        latent = self.encoder(features[:, self.state_dim:])
        return self.decoder(torch.cat((state, latent), dim=1))


METHODS = {
    "jacobian3": "jacobian3",
    "jacobian9": "jacobian9",
    "spring3": "spring3",
    "surface3": "greedy3",
    "surface5": "greedy5",
    "surface3_no_prior": "greedy3",
    "surface5_no_prior": "greedy5",
    "link_spring": "applied",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prior-input", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32])
    args = parser.parse_args()
    torch.set_num_threads(2)
    input_dir, method = args.input, args.method
    with np.load(input_dir / "data.npz") as data:
        state, split = data["state"], data["split"]
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        clip = data["clip"]
        input_width = data["greedy3"].shape[1]
        if method not in ("surface3", "surface5", "link_spring"):
            force = data[METHODS[method]]

    if method in ("surface3", "surface5"):
        with np.load(args.prior_input / "data.npz") as data:
            np.testing.assert_array_equal(clip, data["clip"])
            np.testing.assert_array_equal(split, data["split"])
            np.testing.assert_array_equal(state, data["state"])
            np.testing.assert_array_equal(target, np.c_[data["delta_q"], data["delta_root_1"],
                                                        data["delta_root_5"]])
            force = data[METHODS[method]]

    train, test = split == 0, split == 2
    if method == "link_spring":
        with np.load(input_dir / "link_spring" / "features.npz") as data:
            force = data["applied"].reshape(len(state), -1)
        floor = .1
    else:
        meta = json.loads((input_dir / "manifest.json").read_text())
        floor = np.tile([.1] * len(meta["bodies"]) + [.01] * 3 + [.1] * 4, 5)
    force = (force - force[train].mean(0)) / np.maximum(force[train].std(0), floor)
    force = np.pad(force, ((0, 0), (0, input_width - force.shape[1])))
    state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
    x = np.c_[state, force].astype(np.float32)

    output = input_dir / "force_latent" / method
    output.mkdir(parents=True, exist_ok=True)
    for dim in args.dims:
        def make_model():
            return ForceLatentDecoder(state.shape[1], force.shape[1], dim, target.shape[1])

        rows = []
        for seed in (11, 22, 33):
            pred, _, epoch = fit(x, target, split, seed, output / f"dim{dim}_{seed}.pt",
                                 model_factory=make_model)
            rows.append(dict(seed=seed, best_epoch=epoch, groups={
                name: metrics(target[test, group], pred[:, group]) for name, group in GROUPS.items()
            }))
        (output / f"dim{dim}.json").write_text(json.dumps(rows, indent=2))
        print(method, dim, flush=True)


if __name__ == "__main__":
    main()
