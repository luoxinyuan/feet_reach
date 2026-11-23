"""Measure motion reconstruction with a free per-transition latent."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.force_mapping_eval.evaluate_v2 import GROUPS, TARGET_SCALE, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with np.load(args.input / "data.npz") as data:
        train, test = data["split"] == 0, data["split"] == 2
        state = data["state"]
        state = (state - state[train].mean(0)) / np.maximum(state[train].std(0), .01)
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        contact_width = data["greedy3"].shape[1]
    if not 0 < args.dim <= contact_width:
        parser.error("--dim must fit in the decoder input")
    train_state, test_state, train_target, test_target = (
        torch.as_tensor(array, device="cuda", dtype=torch.float32)
        for array in (state[train], state[test], target[train] / TARGET_SCALE,
                      target[test] / TARGET_SCALE)
    )
    output = args.input / "latent_oracle" / f"dim{args.dim}"
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        decoder = nn.Sequential(
            nn.Linear(state.shape[1] + contact_width, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, target.shape[1]),
        ).cuda()

        def decode(current_state, latent):
            padded = nn.functional.pad(latent, (0, contact_width - args.dim))
            return decoder(torch.cat((current_state, padded), dim=1))

        latent = nn.Embedding(len(train_state), args.dim, sparse=True, device="cuda")
        nn.init.normal_(latent.weight, std=.01)
        decoder_opt = torch.optim.AdamW(decoder.parameters(), lr=.001, weight_decay=.0001)
        latent_opt = torch.optim.SparseAdam(latent.parameters(), lr=.01)
        for _ in range(100):
            for rows in torch.randperm(len(train_state), device="cuda").split(2048):
                code = latent(rows)
                loss = (decode(train_state[rows], code) - train_target[rows]).square().mean()
                loss = loss + .001 * code.square().mean()
                decoder_opt.zero_grad(set_to_none=True)
                latent_opt.zero_grad(set_to_none=True)
                loss.backward()
                decoder_opt.step()
                latent_opt.step()

        decoder.eval().requires_grad_(False)
        seed_offset = 1000
        report = {"seed": seed}
        for name, fitted_width in (("all_targets", target.shape[1]), ("joint_targets", 29)):
            torch.manual_seed(seed + seed_offset)
            prediction = []
            for current_state, motion in zip(test_state.split(2048), test_target.split(2048)):
                code = nn.Parameter(.01 * torch.randn(len(motion), args.dim, device="cuda"))
                optimizer = torch.optim.Adam([code], lr=.02)
                for _ in range(200):
                    error = decode(current_state, code)[:, :fitted_width] - motion[:, :fitted_width]
                    loss = error.square().mean() + .001 * code.square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    prediction.append(decode(current_state, code).cpu().numpy())
            prediction = np.concatenate(prediction) * TARGET_SCALE
            report[name] = {
                key: metrics(target[test, sl], prediction[:, sl]) for key, sl in GROUPS.items()
            }
        reports.append(report)
        torch.save(decoder.state_dict(), output / f"decoder_{seed}.pt")
        print(args.dim, seed, report, flush=True)
    (output / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
