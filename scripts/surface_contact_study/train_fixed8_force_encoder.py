"""Distill the free-motion latent using only causal fixed8 force histories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from active_adaptation.utils.fixed8_force_encoder import Fixed8ForceEncoder
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, TARGET_SCALE, metrics
from scripts.surface_contact_study.latent_to_force import fit_codes, motion_model


HISTORY = (0, 1, 2, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root, out = args.input, args.output
    out.mkdir(parents=True, exist_ok=True)
    reference = root / "latent_oracle" / "dim32" / "decoder_11.pt"
    generator = root / "fixed8_learned_force" / (
        "angle0_smooth0.001_search_inner100_groups_width384_h2_compact_probes12_retain_adaptive_calibrated"
    ) / "seed11"
    with np.load(root / "data.npz") as data:
        state, split, step = (data[k] for k in ("state", "split", "step"))
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
        width = data["greedy3"].shape[1]
    with np.load(generator / "aligned_forces.npz") as data:
        force = data["force"]
    with np.load(generator / "test_predictions.npz") as data:
        np.testing.assert_array_equal(force[split == 2], data["force"])
    cache_meta = json.loads((args.force_cache / "manifest.json").read_text())
    assert cache_meta["version"] == "fixed8-aligned-observed-v1"
    assert cache_meta["solver_sha256"] == hashlib.sha256((generator / "solver.pt").read_bytes()).hexdigest()
    assert cache_meta["points"] == json.loads((root / "masked_spring" / "points.json").read_text())
    if force.shape != (len(state), 8, 3):
        raise ValueError("Expected one 8-point force record per benchmark transition")
    frames = np.arange(len(state))
    history = []
    for offset in HISTORY:
        observed = force[np.maximum(frames - offset, 0)].copy()
        observed[step < offset] = 0.
        history.append(observed.reshape(len(force), 24))
    train, valid, test = split == 0, split == 1, split == 2
    cuda = lambda values: torch.as_tensor(values, dtype=torch.float32, device="cuda")
    x = cuda(np.concatenate(history, axis=1))
    s = cuda((state - state[train].mean(0)) / np.maximum(state[train].std(0), .01))
    y = cuda(target / TARGET_SCALE)
    decoder = motion_model(s.shape[1] + width, y.shape[1])
    decoder.load_state_dict(torch.load(reference, map_location="cpu", weights_only=True))
    decoder.eval().requires_grad_(False)
    tr, va, te = (torch.as_tensor(np.flatnonzero(mask), device="cuda") for mask in (train, valid, test))
    # A single fixed motion decoder establishes the coordinates for every fitted code.
    codes = fit_codes(decoder, s[tr], y[tr], width, args.seed)
    code_mean = codes.mean(0)
    code_scale = codes.std(0, correction=0).clamp_min(.01)
    code_target = torch.zeros((len(s), 32), device="cuda")
    code_target[tr] = (codes - code_mean) / code_scale
    rms = y[tr].square().mean(0).sqrt().clamp_min(.025)
    loss_weight = torch.ones(y.shape[1], device="cuda")
    for group in GROUPS.values():
        loss_weight[group] = 1. / (rms[group].square().sum().sqrt() * np.sqrt(len(GROUPS)))
    mean_force, scale_force = x[tr].mean(0), x[tr].std(0, correction=0).clamp_min(.10)
    reports, models = [], []
    print(f"Fitted {len(tr):,} training codes, with {len(va):,} validation and {len(te):,} test frames",
          flush=True)

    def predict(model, rows, force_history):
        z = model.motion_code(model(force_history))
        return decoder(torch.cat((s[rows], nn.functional.pad(z, (0, width - 32))), 1))

    def score(model, rows, inputs):
        with torch.no_grad():
            prediction = predict(model, rows, inputs).cpu().numpy() * TARGET_SCALE
        groups = {name: metrics(target[rows.cpu().numpy(), sl], prediction[:, sl])
                  for name, sl in GROUPS.items()}
        return dict(nrmse=float(np.mean([v["nrmse"] for v in groups.values()])), groups=groups)

    for alignment in (0., .02, .08):
        torch.manual_seed(args.seed)
        encoder = Fixed8ForceEncoder(x.shape[1]).cuda()
        for name, value in (("force_mean", mean_force), ("force_scale", scale_force),
                            ("code_mean", code_mean), ("code_scale", code_scale)):
            getattr(encoder, name).copy_(value)
        opt = torch.optim.AdamW(encoder.parameters(), lr=.001, weight_decay=.0001)
        best, weights, stale = float("inf"), None, 0
        for epoch in range(125):
            for rows in tr[torch.randperm(len(tr), device="cuda")].split(2048):
                latent = encoder(x[rows])
                code = encoder.motion_code(latent)
                prediction = decoder(torch.cat((s[rows], nn.functional.pad(code, (0, width - 32))), 1))
                motion_loss = ((prediction - y[rows]) * loss_weight).square().sum(-1).mean()
                nonzero = (x[rows].abs().amax(-1) > 1e-6).float()
                align_loss = ((latent - code_target[rows]).square().mean(-1) * nonzero).sum()
                loss = motion_loss + alignment * align_loss / nonzero.sum().clamp_min(1.)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            val_score = score(encoder, va, x[va])["nrmse"]
            if val_score < best - 1e-5:
                best, weights, stale = val_score, {k: v.detach().clone() for k, v in encoder.state_dict().items()}, 0
                best_epoch = epoch + 1
            else:
                stale += 1
                if stale >= 18:
                    break
        reports.append(dict(alignment_weight=alignment, validation_nrmse=best, epoch=best_epoch))
        models.append(weights)
        print(f"alignment={alignment:g} val={best:.4f} epoch={best_epoch}", flush=True)
    # This run specifically tests alignment. The motion-only branch is a control.
    pick = min((i for i, r in enumerate(reports) if r["alignment_weight"] > 0),
               key=lambda i: reports[i]["validation_nrmse"])
    encoder = Fixed8ForceEncoder(x.shape[1]).cuda()
    encoder.load_state_dict(models[pick])
    encoder.eval().requires_grad_(False)
    controls = dict(clean=score(encoder, te, x[te]), zero_force=score(encoder, te, torch.zeros_like(x[te])))
    order = np.random.default_rng(args.seed).permutation(len(te))
    controls["shuffled_force"] = score(encoder, te, x[te[order]])
    result = dict(variants=reports, selected=reports[pick], test=controls,
                  frames={k: len(r) for k, r in (("train", tr), ("validation", va), ("test", te))})
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(dict(version="fixed8-force-encoder-v1", history_offsets=list(HISTORY),
                    points=cache_meta["points"], solver_sha256=cache_meta["solver_sha256"],
                    state_dict={k: v.cpu() for k, v in encoder.state_dict().items()}),
               out / "encoder.pt")
    print("FIXED8_ENCODER_READY", json.dumps(result["selected"]),
          {k: v["nrmse"] for k, v in controls.items()}, flush=True)


if __name__ == "__main__":
    main()
