"""Select an 8KP force bottleneck by validation, then compare on held-out clips."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from active_adaptation.utils.fixed8_force_encoder import (
    FIXED8_FORCE_HISTORY_OFFSETS as HISTORY, Fixed8ForceEncoder, Fixed8TemporalForceEncoder,
)
from scripts.force_mapping_eval.evaluate_v2 import GROUPS, TARGET_SCALE, metrics


def motion_readout(input_width, hidden, outputs):
    return nn.Sequential(nn.Linear(input_width, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ConditionalReadout(nn.Module):
    """Predict motion with learned products between posture and force features."""

    def __init__(self, state_width, force_width, hidden, outputs):
        super().__init__()
        self.pose = nn.Linear(state_width, hidden)
        self.force = nn.Linear(force_width, hidden, bias=False)
        self.readout = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

    def forward(self, state, force):
        pose = self.pose(state)
        return self.readout(pose + self.force(force) * (1. + pose.tanh()))


class MotionProbe(nn.Module):
    def __init__(self, state_width, force_width, hidden, outputs, encoder=None, structure="joint"):
        super().__init__()
        self.encoder = encoder
        self.structure = structure
        self.root_channels = 12 if encoder is not None else force_width
        if structure == "split":
            self.root_readout = motion_readout(state_width + self.root_channels, hidden, 12)
            self.joint_readout = motion_readout(state_width + force_width, hidden, 29)
        elif structure == "conditional":
            self.readout = ConditionalReadout(state_width, force_width, hidden, outputs)
        elif structure == "split_conditional":
            self.root_readout = ConditionalReadout(state_width, self.root_channels, hidden, 12)
            self.joint_readout = ConditionalReadout(state_width, force_width, hidden, 29)
        else:
            self.readout = motion_readout(state_width + force_width, hidden, outputs)

    def forward(self, state, features, loaded):
        if isinstance(self.encoder, Fixed8TemporalForceEncoder):
            features = torch.where(loaded, self.encoder.encode(features), 0.)
        elif self.encoder is not None:
            features = self.encoder(features)
        root = features[:, :self.root_channels]
        if self.structure == "split":
            return torch.cat((self.joint_readout(torch.cat((state, features), dim=-1)),
                              self.root_readout(torch.cat((state, root), dim=-1))), dim=-1)
        if self.structure == "split_conditional":
            return torch.cat((self.joint_readout(state, features), self.root_readout(state, root)), dim=-1)
        if self.structure == "conditional":
            return self.readout(state, features)
        return self.readout(torch.cat((state, features), dim=-1))


def histories(force, clip, step):
    frames = np.arange(len(force))
    blocks = []
    for offset in HISTORY:
        prev = np.maximum(frames - offset, 0)
        valid = (clip[prev] == clip) & (step - step[prev] == offset)
        block = force[prev].reshape(len(force), 24)
        block[~valid] = 0.
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def mirror_training(input_dir, state, target, history):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    names = json.loads((Path(manifest["source"]) / "manifest.json").read_text())["joint_names"]
    asset = Path(__file__).resolve().parents[2] / "active_adaptation/assets/humanoid.py"
    symmetry = next(node.value for node in ast.walk(ast.parse(asset.read_text()))
                    if isinstance(node, ast.keyword) and node.arg == "joint_symmetry_mapping"
                    and isinstance(node.value, ast.Call)
                    and set(names).issubset(set(ast.literal_eval(node.value.args[0])) |
                                            {v[1] for v in ast.literal_eval(node.value.args[0]).values()}))
    mapping = ast.literal_eval(symmetry.args[0])
    mapping.update({other: (sign, joint) for joint, (sign, other) in list(mapping.items())})
    order = [names.index(mapping[joint][1]) for joint in names]
    sign = np.asarray([mapping[joint][0] for joint in names], dtype=np.float32)
    reflect = np.asarray([1., -1., 1.], dtype=np.float32)
    force = history.reshape(-1, len(HISTORY), 8, 3)[:, :, [0, 2, 1, 3, 5, 4, 7, 6]] * reflect
    mirrored_state = np.c_[state[:, order] * sign, state[:, 29:32] * reflect, state[:, 32:]]
    mirrored_target = np.c_[target[:, order] * sign, target[:, 29:] * np.tile(np.r_[reflect, -reflect], 2)]
    return mirrored_state, mirrored_target, force.reshape(history.shape), order, sign


def scores(target, prediction):
    groups = {name: metrics(target[:, sl], prediction[:, sl]) for name, sl in GROUPS.items()}
    return dict(nrmse=float(np.mean([value["nrmse"] for value in groups.values()])), groups=groups)


def paired_ci(target, predictions, raw_predictions, clip):
    _, clip_index = np.unique(clip, return_inverse=True)
    count = int(clip_index.max()) + 1
    # Average normalized error only after reconstructing each seed's five group scores.
    blocks = np.empty((3, len(predictions), len(GROUPS), count), dtype=np.float64)
    for index, sl in enumerate(GROUPS.values()):
        for seed, (pred, raw) in enumerate(zip(predictions, raw_predictions)):
            per_frame = ((pred[:, sl] - target[:, sl]) ** 2, (raw[:, sl] - target[:, sl]) ** 2,
                         target[:, sl] ** 2)
            for field, values in enumerate(per_frame):
                blocks[field, seed, index] = np.bincount(clip_index, values.sum(1), minlength=count)
    rng = np.random.default_rng(902)
    weights = rng.multinomial(count, np.full(count, 1. / count), size=2000).T
    sse, raw_sse, energy = blocks @ weights
    energy = np.maximum(energy, 1e-15)
    delta = (np.sqrt(sse / energy) - np.sqrt(raw_sse / energy)).mean(axis=(0, 1))
    return np.quantile(delta, [.025, .975]).tolist()


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    with np.load(args.input / "data.npz") as data:
        state, split, step, clip = (data[key] for key in ("state", "split", "step", "clip"))
        target = np.c_[data["delta_q"], data["delta_root_1"], data["delta_root_5"]]
    with np.load(args.generator / "aligned_forces.npz") as data:
        force = data["force"]
    with np.load(args.generator / "test_predictions.npz") as data:
        np.testing.assert_array_equal(force[split == 2], data["force"])
    meta = json.loads((args.force_cache / "manifest.json").read_text())
    points = json.loads((args.input / "masked_spring" / "points.json").read_text())
    assert meta["version"] == "fixed8-aligned-observed-v1" and meta["points"] == points
    assert meta["solver_sha256"] == hashlib.sha256((args.generator / "solver.pt").read_bytes()).hexdigest()
    assert force.shape == (len(state), 8, 3)
    assert all(len(np.unique(split[clip == value])) == 1 for value in np.unique(clip))
    assert [item["body"] for item in points] == [
        "pelvis", "left_hand_mimic", "right_hand_mimic", "torso_link", "left_shoulder_roll_link",
        "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link"]
    history = histories(force, clip, step)
    x = torch.from_numpy(history).to(device)
    loaded = x.abs().amax(-1, keepdim=True) > 1e-6
    indices = np.arange(len(state))
    horizons, horizon_masks = [], []
    for horizon in (5, 10):
        endpoint = np.minimum(indices + horizon, len(state) - 1)
        horizon_masks.append((clip[endpoint] == clip) & (step[endpoint] - step == horizon))
        horizons.append(state[endpoint, :29] - state[:, :29])
    extension = torch.from_numpy(np.concatenate(horizons, axis=1) / .02).to(device)
    extension_mask = torch.from_numpy(np.repeat(np.stack(horizon_masks, axis=1), 29, axis=1)).to(device)
    state_source = state
    state = torch.from_numpy(state_source).to(device)
    y = torch.from_numpy(target / TARGET_SCALE).to(device)
    tr, va, te = [torch.as_tensor(np.flatnonzero(split == fold), device=device) for fold in range(3)]

    def normalize(values, minimum_scale, rows):
        mean = values[rows].mean(0)
        scale = values[rows].std(0, correction=0).clamp_min(minimum_scale)
        return (values - mean) / scale, mean, scale

    rms = y[tr].square().mean(0).sqrt().clamp_min(.025)
    scale = torch.ones(len(TARGET_SCALE), device=device)
    for sl in GROUPS.values():
        scale[sl] = 1. / (rms[sl].norm() * np.sqrt(len(GROUPS)))
    val_target = target[split == 1]
    test_target = target[split == 2]
    n_outputs = len(TARGET_SCALE)

    def fit(name, seed):
        torch.manual_seed(seed)
        mirror = name.endswith(":mirror")
        model_kind, hidden_str, *flags = name.removesuffix(":mirror").split(":")
        structure = "joint"
        if flags and flags[-1] in ("split", "conditional", "split_conditional"):
            structure = flags.pop()
        hidden = int(hidden_str)
        aux_weight = float(flags[0]) if flags else 0.
        extra_weight = float(flags[1]) if len(flags) > 1 else 0.
        poses, force_history, labels, active, train, future, future_mask = (
            state, x, y, loaded, tr, extension, extension_mask)
        if mirror:
            source_rows = np.flatnonzero(split == 0)
            mirrored_state, mirrored_target, mirrored_force, order, signs = mirror_training(
                args.input, state_source[source_rows], target[source_rows], history[source_rows])
            poses = torch.cat((poses, torch.from_numpy(mirrored_state).to(device)))
            labels = torch.cat((labels, torch.from_numpy(mirrored_target / TARGET_SCALE).to(device)))
            force_history = torch.cat((force_history, torch.from_numpy(mirrored_force).to(device)))
            active = force_history.abs().amax(-1, keepdim=True) > 1e-6
            train = torch.cat((tr, torch.arange(len(tr), device=device) + len(x)))
            future_order = order + [i + 29 for i in order]
            future = torch.cat((future, future[tr][:, future_order] *
                                torch.from_numpy(np.tile(signs, 2)).to(device)))
            future_mask = torch.cat((future_mask, future_mask[tr][:, future_order]))
        poses, _, _ = normalize(poses, .01, train)
        raw, raw_mean, raw_scale = normalize(force_history, .10, train)
        encoder = None
        features = raw
        if model_kind == "cartesian_latent":
            encoder = Fixed8ForceEncoder(x.shape[1], hidden).to(device)
            encoder.force_mean.copy_(raw_mean)
            encoder.force_scale.copy_(raw_scale)
            features = force_history
        elif model_kind != "raw":
            encoder = Fixed8TemporalForceEncoder(len(HISTORY), hidden, model_kind).to(device)
            features = encoder.lift(force_history)
            _, mean, std = normalize(features, .04, train)
            encoder.feature_mean.copy_(mean)
            encoder.feature_scale.copy_(std)
        outputs = n_outputs + (extension.shape[1] if extra_weight else 0)
        probe = MotionProbe(state.shape[-1], x.shape[-1] if encoder is None else 32,
                            hidden, outputs, encoder, structure).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=.001, weight_decay=.0001)
        best, weights, age = float("inf"), None, 0
        index = 29  # The first 12 latent channels can optionally be grounded in root displacement.
        for epoch in range(1, args.epochs + 1):
            order = train[torch.randperm(len(train), device=device)]
            for rows in order.split(2048):
                preds = probe(poses[rows], features[rows], active[rows])
                loss = ((preds[:, :n_outputs] - labels[rows]) * scale).square().sum(-1).mean()
                if extra_weight:
                    future_error = ((preds[:, n_outputs:] - future[rows]) * future_mask[rows]).square()
                    loss = loss + extra_weight * future_error.mean()
                if aux_weight:
                    latent = encoder.encode(features[rows])
                    root = labels[rows, index:] / rms[index:]
                    loss = loss + aux_weight * ((latent[:, :root.shape[1]] - root).square().mean(1)
                                               * active[rows].squeeze(1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = probe(poses[va], features[va], active[va])[:, :n_outputs].cpu().numpy() * TARGET_SCALE
            validation = scores(val_target, pred)["nrmse"]
            if validation < best - 1e-5:
                best, age, best_epoch = validation, 0, epoch
                weights = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            else:
                age += 1
            if age >= 15:
                break
        probe.load_state_dict(weights)
        details = dict(name=name, seed=seed, parameters=sum(p.numel() for p in probe.parameters()),
                       validation_nrmse=best, epoch=best_epoch)
        print(json.dumps(details), flush=True)
        return probe, poses, features, details

    args.output.mkdir(parents=True, exist_ok=True)
    if args.search:
        records = [fit(name, seed)[3] for name in args.search for seed in args.seeds]
        (args.output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
        return

    records, predictions = {}, {}
    encoder_weights = []
    names = [*args.raw, args.candidate]
    for name in names:
        records[name], predictions[name] = [], []
        for seed in args.seeds:
            model, poses, features, details = fit(name, seed)
            with torch.no_grad():
                pred = model(poses[te], features[te], loaded[te])[:, :n_outputs].cpu().numpy() * TARGET_SCALE
            details["test"] = scores(test_target, pred)
            records[name].append(details)
            predictions[name].append(pred)
            print(f"test {name} seed={seed}: {details['test']['nrmse']:.4f}", flush=True)
            if name == args.candidate:
                encoder_weights.append({k: v.cpu() for k, v in model.encoder.state_dict().items()})

    raw_best = min(args.raw, key=lambda name: np.mean([r["validation_nrmse"] for r in records[name]]))
    comparisons = {}
    for raw_name in args.raw:
        differences = [m["test"]["nrmse"] - r["test"]["nrmse"]
                       for m, r in zip(records[args.candidate], records[raw_name])]
        ci = paired_ci(test_target, predictions[args.candidate], predictions[raw_name], clip[split == 2])
        comparisons[raw_name] = dict(paired_seed_deltas=differences, paired_clip_bootstrap95=ci,
                                     improvement_confirmed=max(differences) < 0. and ci[1] < 0.)
    passed = all(result["improvement_confirmed"] for result in comparisons.values())
    exported = ""
    if passed:
        # The deployed seed is selected by validation alone.
        pick = int(np.argmin([r["validation_nrmse"] for r in records[args.candidate]]))
        features, hidden, *_ = args.candidate.split(":")
        encoder_path = args.output / "encoder.pt"
        torch.save(dict(version="fixed8-force-encoder-v2", history_offsets=list(HISTORY), points=points,
                        solver_sha256=meta["solver_sha256"], features=features, hidden=int(hidden),
                        state_dict=encoder_weights[pick]), encoder_path)
        exported = str(encoder_path)
    report = dict(candidates=records, best_raw_by_validation=raw_best, candidate=args.candidate,
                  heldout_clips=int(np.unique(clip[split == 2]).size), seeds=args.seeds,
                  frames=dict(train=len(tr), validation=len(va), test=len(te)), history_offsets=list(HISTORY),
                  comparisons=comparisons,
                  improvement_confirmed=passed, encoder_path=exported)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"test candidate-minus-raw: {json.dumps(comparisons)}; pass={passed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--force-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--search", nargs="+")
    parser.add_argument("--raw", nargs="+", default=["raw:256", "raw:384"])
    parser.add_argument("--candidate")
    args = parser.parse_args()
    if not args.search and (args.candidate is None or args.candidate.startswith("raw:")):
        parser.error("Select a temporal candidate by validation before running the held-out comparison.")
    run(args)
