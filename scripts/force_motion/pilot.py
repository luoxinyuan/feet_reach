"""Offline force-to-motion pilot on ComplianceMix G1 recordings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from active_adaptation.utils.body_guidance import limit_force, slew_force
from active_adaptation.utils.surface_fk import SurfaceFK, multiply
from active_adaptation.utils.surface_guidance import SurfaceSampler, quat_apply


HISTORY = 17
FUTURE = 25
CONTACTS = 5
DT = 0.02
FRAMES = HISTORY + FUTURE
STATE_DIM = 35
FIELDS = ("root_pos_w", "root_quat_w", "joint_pos")


def recording_key(row, group_bones_takes):
    name = re.sub(r"_M(?=\.csv$)", "", row["source_name"])
    if group_bones_takes and row["dataset"] == "boneseed":
        name = re.sub(r"__A\d+(?=\.csv$)", "", name)
    return f"{row['dataset']}/{name}"


def split_recording(key, seed):
    digest = hashlib.sha256(f"{seed}/{key}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "little") % 100
    return "test" if bucket < 15 else "validation" if bucket < 25 else "train"


def load_windows(path, seed, windows_per_clip, clip_limit, group_bones_takes):
    rng = np.random.default_rng(seed)
    records = []
    windows = []
    names = None
    for category in sorted((path / "train").iterdir()):
        source = category / "_tensordict"
        meta = json.loads((source / "meta.json").read_text())
        clip_names = json.loads((category / "meta_motion.json").read_text())["joint_names"]
        if names is not None and names != clip_names:
            raise ValueError(f"Joint order differs in {category}")
        names = clip_names
        arrays = [
            np.memmap(source / f"{field}.memmap", mode="r", shape=tuple(meta[field]["shape"]), dtype=np.float32)
            for field in FIELDS
        ]
        provenance = json.loads((category / "provenance.json").read_text())
        for split in ("train", "validation", "test"):
            rows = []
            for row in provenance:
                if row["dataset"] not in {"boneseed", "lafan"}:
                    continue
                if row["end"] - row["start"] < FRAMES:
                    continue
                key = recording_key(row, group_bones_takes)
                if split_recording(key, seed) != split:
                    continue
                rows.append((key, row))
            if clip_limit:
                limit = clip_limit if split == "train" else max(12, clip_limit // 4)
                rows = [rows[i] for i in rng.permutation(len(rows))[:limit]]
            for key, row in rows:
                starts = rng.choice(
                    row["end"] - row["start"] - FRAMES + 1,
                    size=min(windows_per_clip, row["end"] - row["start"] - FRAMES + 1),
                    replace=False,
                )
                for offset in starts:
                    start = int(row["start"] + offset)
                    windows.append(np.concatenate([a[start:start + FRAMES] for a in arrays], axis=-1))
                    records.append((split, category.name, key, int(row["motion_id"]), start))
    return np.stack(windows), records, names


def inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    half = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)) / 2
    return torch.stack((half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), -1)


def rotation_vector(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    s = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., :1])
    return q[..., 1:] * angle / s.clamp_min(1e-8)


def rotation_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat(((angle / 2).cos(), v * (torch.sinc(angle / (2 * torch.pi)) / 2)), -1)


@torch.no_grad()
def encode(windows, fk, surface, device):
    states, forces, contacts, counts, violations = [], [], [], [], []
    excluded = torch.tensor(
        [any(part in name for part in ("hip", "knee", "ankle")) for name in fk.bodies],
        device=device,
    )
    for raw in torch.as_tensor(windows, device=device).split(256):
        n = len(raw)
        p, r, q = raw[..., :3], F.normalize(raw[..., 3:7], dim=-1), raw[..., 7:]
        heading = yaw_quat(r[:, HISTORY - 1])
        heading_inv = inverse(heading)
        anchor = p[:, HISTORY - 1].clone()
        anchor[:, 2] = 0
        frame = heading_inv[:, None].expand_as(r)
        state = torch.cat(
            (quat_apply(frame, p - anchor[:, None]), rotation_vector(multiply(frame, r)), q), -1
        )
        positions, rotations = fk(q.flatten(0, 1), p.flatten(0, 1), r.flatten(0, 1))
        positions = positions.reshape(n, FRAMES, -1, 3)
        rotations = rotations.reshape(n, FRAMES, -1, 4)
        row = torch.arange(n, device=device)[:, None]
        body, tri, bary = surface.sample((n, 128))
        local = (tri * bary[..., None]).sum(-2)
        first_point = positions[row, 0, body] + quat_apply(rotations[row, 0, body], local)
        valid = (~excluded[body]) & (first_point[..., 2] >= 0.06)
        indices = valid.int().argsort(dim=-1, descending=True, stable=True)[:, :CONTACTS]
        body, local = body[row, indices], local[row, indices]
        count = torch.minimum(torch.randint(1, CONTACTS + 1, (n,), device=device), valid.sum(-1))
        active = torch.arange(CONTACTS, device=device)[None] < count[:, None]
        frame_ids = torch.arange(FRAMES, device=device)[None, :, None]
        points = positions[row[:, None], frame_ids, body[:, None]] + quat_apply(
            rotations[row[:, None], frame_ids, body[:, None]],
            local[:, None].expand(-1, FRAMES, -1, -1),
        )
        # Same next-reference Cartesian spring and limits as ObservableSurfaceCommand.
        desired = 80.0 * (points[:, 1:] - points[:, :-1]) * active[:, None, :, None]
        desired = limit_force(desired.flatten(0, 1)).reshape_as(desired)
        force = torch.zeros_like(desired[:, 0])
        sequence = []
        for target in desired.unbind(1):
            # Four 200 Hz physics substeps with the same held target.
            for _ in range(4):
                force = slew_force(force, target, DT / 4)
            sequence.append(force)
        force = torch.stack(sequence, 1)
        norms = force.norm(dim=-1)
        initial_zero = torch.zeros_like(force[:, :1])
        slew = torch.diff(torch.cat((initial_zero, force), 1), dim=1).norm(dim=-1) / DT
        violations.append(torch.stack((norms.amax(), norms.sum(-1).amax(), slew.amax())))
        force = quat_apply(heading_inv[:, None, None].expand(*force.shape[:-1], 4), force)
        visible = (norms > 1e-5) & active[:, None]
        info = torch.cat((F.one_hot(body, len(fk.bodies)), local * 10), -1)
        force = torch.cat(
            (info[:, None].expand(-1, FRAMES - 1, -1, -1), force / 3, visible[..., None]), -1
        ) * visible[..., None]
        states.append(state)
        forces.append(force)
        contacts.append((body, local))
        counts.append(count)
    body, local = (torch.cat(pieces) for pieces in zip(*contacts))
    return torch.cat(states), torch.cat(forces), body, local, torch.cat(counts), torch.stack(violations).amax(0)


class MotionPredictor(nn.Module):
    """Frame-aligned history and pooled contact tokens with a temporal backbone."""

    def __init__(self, force_dim, width=96):
        super().__init__()
        self.state = nn.Linear(STATE_DIM + 1, width)
        self.contact = nn.Sequential(nn.Linear(force_dim, 48), nn.GELU(), nn.Linear(48, width))
        self.position = nn.Parameter(torch.randn(FRAMES - 1, width) * 0.02)
        self.noise = nn.Linear(STATE_DIM + 1, width)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, STATE_DIM))

    def forward(self, history, force, noisy, tau):
        n = len(history)
        state = F.pad(history, (0, 1, 0, FUTURE - 1))
        state[:, :HISTORY, -1] = 1
        embedded = self.state(state) + self.position
        contact = self.contact(force) * force[..., -1:]
        embedded = embedded + contact.sum(-2) / CONTACTS
        value = self.noise(torch.cat((noisy, tau[:, None, None].expand(n, FUTURE, 1)), -1))
        # Force at t labels the transition t -> t+1, including the first target.
        embedded = torch.cat((embedded[:, :HISTORY - 1], embedded[:, HISTORY - 1:] + value), 1)
        return self.output(self.blocks(embedded)[:, HISTORY - 1:])


def force_condition(force, mode):
    if mode == "history":
        return torch.zeros_like(force)
    selected = force.clone()
    if mode in {"causal", "delayed"}:
        last = HISTORY if mode == "causal" else HISTORY - 1
        selected[:, last:] = 0
    if mode == "metadata":
        selected[..., -4:-1] = 0
    return selected


def predict(model, history, force, *, noise=None, steps=8):
    if noise is None:
        x = history.new_zeros(len(history), FUTURE, STATE_DIM)
        return model(history, force, x, x.new_zeros(len(x)))
    x = noise
    for step in range(steps):
        tau = x.new_full((len(x),), step / steps)
        x = x + model(history, force, x, tau) / steps
    return x


def train_model(mode, flow, history, target, force, train, valid, args):
    torch.manual_seed(args.seed)
    model = MotionPredictor(force.shape[-1]).to(force.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    noise_generator = torch.Generator(device=force.device).manual_seed(args.seed + 73)
    valid_noise = torch.randn(target[valid].shape, device=force.device, generator=noise_generator)
    epochs = args.flow_epochs if flow else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=8e-5)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train[torch.randperm(len(train), device=train.device)].split(args.batch):
            clean = target[batch]
            if flow:
                eps = torch.randn_like(clean)
                tau = torch.rand(len(batch), device=force.device)
                noisy = torch.lerp(eps, clean, tau[:, None, None])
                label = clean - eps
            else:
                noisy = torch.zeros_like(clean)
                tau = clean.new_zeros(len(clean))
                label = clean
            prediction = model(history[batch], force_condition(force[batch], mode), noisy, tau)
            loss = F.mse_loss(prediction, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            result = []
            for ids, eps in zip(valid.split(args.batch), valid_noise.split(args.batch)):
                prediction = predict(model, history[ids], force_condition(force[ids], mode), noise=eps if flow else None)
                result.append((prediction - target[ids]).square().mean((1, 2)))
            score = torch.cat(result).mean().item()
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == epochs:
            print(f"{mode}{'-flow' if flow else ''} epoch {epoch+1}/{epochs} "
                  f"loss={torch.stack(losses).mean():.4f} val={score:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, {"epoch": best_epoch, "validation_scaled_mse": best_score}


def shuffled_indices(records, test, counts, seed):
    rng = np.random.default_rng(seed + 791)
    groups = {}
    test_ids = test.tolist()
    for idx in test_ids:
        _, category, key, _, _ = records[idx]
        groups.setdefault((category, int(counts[idx])), []).append((idx, key))
    donors = []
    for idx in test_ids:
        _, category, key, _, _ = records[idx]
        candidates = [j for j, other in groups[(category, int(counts[idx]))] if other != key]
        if not candidates:
            candidates = [j for j in test_ids if records[j][2] != key and counts[j] == counts[idx]]
        if not candidates and counts[idx] == 0:
            # The zero-contact condition is identical for every donor.
            candidates = [idx]
        if not candidates:
            raise ValueError("No held-out donor with the same contact count and a different recording")
        donors.append(rng.choice(candidates))
    return donors


@torch.no_grad()
def metrics(fk, predicted, actual):
    rotation = rotation_quat(predicted[..., 3:6].flatten(0, 1))
    rotation_gt = rotation_quat(actual[..., 3:6].flatten(0, 1))
    q_rel = multiply(rotation, inverse(rotation_gt))
    pred, _ = fk(predicted[..., 6:].flatten(0, 1), predicted[..., :3].flatten(0, 1), rotation)
    ref, _ = fk(actual[..., 6:].flatten(0, 1), actual[..., :3].flatten(0, 1), rotation_gt)
    error = (pred - ref).norm(dim=-1).reshape(len(predicted), FUTURE, -1) * 100
    wrist = [fk.bodies.index(f"{side}_wrist_yaw_link") for side in ("left", "right")]
    feet = [fk.bodies.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    root = (predicted[..., :3] - actual[..., :3]).norm(dim=-1) * 100
    return {
        "joint_mae_rad": (predicted[..., 6:] - actual[..., 6:]).abs().mean((1, 2)),
        "root_cm": root.mean(-1),
        "root_endpoint_cm": root[:, -1],
        "root_rotation_deg": rotation_vector(q_rel).norm(dim=-1).reshape(len(predicted), FUTURE).mean(-1) * (180 / torch.pi),
        "body_fk_cm": error.mean((1, 2)),
        "body_fk_first_100ms_cm": error[:, :5].mean((1, 2)),
        "wrist_fk_cm": error[..., wrist].mean((1, 2)),
        "foot_fk_cm": error[..., feet].mean((1, 2)),
    }


@torch.no_grad()
def reversal_response(fk, forecasts, force, body, local, mode, batch):
    """Project the paired-versus-reversed displacement onto the original force."""
    if mode == "causal":
        # Hold the last observed vector when measuring near-term movement.
        prompt = force[:, HISTORY - 1:HISTORY, :, -4:-1].expand(-1, 5, -1, -1)
    elif mode == "delayed":
        prompt = force[:, HISTORY - 2:HISTORY - 1, :, -4:-1].expand(-1, 5, -1, -1)
    else:
        prompt = force[:, HISTORY - 1:, :, -4:-1]
    points = []
    for variant in ("paired", "reversed"):
        sequence = []
        for motion, ids, offset in zip(forecasts[variant].split(batch), body.split(batch), local.split(batch)):
            motion = motion[:, :prompt.shape[1]]
            n, horizon = motion.shape[:2]
            motion = motion.flatten(0, 1)
            pos, quat = fk(motion[:, 6:], motion[:, :3], rotation_quat(motion[:, 3:6]))
            rows = torch.arange(len(motion), device=motion.device)[:, None]
            ids = ids.repeat_interleave(horizon, dim=0)
            offset = offset.repeat_interleave(horizon, dim=0)
            sequence.append((pos[rows, ids] + quat_apply(quat[rows, ids], offset)).reshape(n, horizon, CONTACTS, 3))
        points.append(torch.cat(sequence))
    absolute = prompt.norm(dim=-1).sum((1, 2))
    keep = absolute > 1e-4
    projection = ((points[0] - points[1]) * prompt).sum((1, 2, 3))[keep] / absolute[keep]
    return {
        "windows_with_force": int(keep.sum()),
        "mean_projected_displacement_cm": (projection.mean() * 100).item(),
        "fraction_positive": (projection > 0).float().mean().item(),
    }


@torch.no_grad()
def evaluate(model, mode, flow, history, target, force, test, donor, fk, scale, states, batch, seed):
    outputs = {}
    forecasts = {}
    variants = ["paired"] if mode == "history" else ["paired", "shuffled", "zero", "reversed"]
    for variant in variants:
        noise_generator = torch.Generator(device=force.device).manual_seed(seed + 88)
        preds = []
        for ids, other in zip(test.split(batch), donor.split(batch)):
            cond = force_condition(force[other if variant == "shuffled" else ids], mode)
            if variant == "zero":
                cond.zero_()
            if variant == "reversed":
                cond[..., -4:-1] *= -1
            noise = torch.randn(target[ids].shape, device=force.device, generator=noise_generator) if flow else None
            preds.append(predict(model, history[ids], cond, noise=noise))
        forecasts[variant] = torch.cat(preds) * scale + states[test, HISTORY - 1:HISTORY]
        outputs[variant] = metrics(fk, forecasts[variant], states[test, HISTORY:])
    return outputs, forecasts


def summarize(values, categories):
    arrays = {name: value.cpu().numpy() for name, value in values.items()}
    return {
        "all": {name: float(v.mean()) for name, v in arrays.items()},
        "by_category": {
            cat: {name: float(v[categories == cat].mean()) for name, v in arrays.items()}
            for cat in sorted(set(categories))
        },
    }


def bootstrap_improvement(baseline, paired, sources, seed):
    names, index = np.unique(sources, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=(baseline - paired).cpu().numpy())
    draws = rng.integers(len(names), size=(2000, len(names)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/compliance_mix_29h_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--windows-per-clip", type=int, default=2)
    parser.add_argument("--clip-limit", type=int, default=0, help="Maximum train clips per category; 0 uses all")
    parser.add_argument("--group-bones-takes", action="store_true", help="Also group BONES files differing only by an __A identifier")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--flow-epochs", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    windows, records, names = load_windows(
        args.dataset, args.seed, args.windows_per_clip, args.clip_limit, args.group_bones_takes
    )
    device = torch.device(args.device)
    geometry = json.loads((ROOT / "artifacts/ceer_segmented_jacobian/geometry.json").read_text())
    fk = SurfaceFK(geometry, names, device)
    surface = SurfaceSampler(ROOT / "artifacts/surface_compliance/geometry", fk.bodies, device)
    states, force, body, local, counts, budgets = encode(windows, fk, surface, device)
    del windows
    splits = {
        key: torch.tensor([i for i, row in enumerate(records) if row[0] == key], device=device)
        for key in ("train", "validation", "test")
    }
    train, valid, test = (splits[key] for key in ("train", "validation", "test"))
    source_sets = {key: {records[i][2] for i in ids.tolist()} for key, ids in splits.items()}
    assert not (source_sets["train"] & source_sets["test"] or source_sets["train"] & source_sets["validation"]
                or source_sets["test"] & source_sets["validation"])
    assert force.isfinite().all() and states.isfinite().all()
    assert (budgets <= budgets.new_tensor([30.001, 60.001, 150.01])).all(), budgets.tolist()
    counts_cpu = counts.cpu().numpy()
    test_cpu = test.tolist()
    donor = torch.tensor(shuffled_indices(records, test, counts_cpu, args.seed), device=device)
    scales = (states[train, HISTORY:] - states[train, HISTORY - 1:HISTORY]).std((0, 1)).clamp_min(0.015)
    center = states[train, :HISTORY].mean((0, 1))
    spread = states[train, :HISTORY].std((0, 1)).clamp_min(0.025)
    history = (states[:, :HISTORY] - center) / spread
    target = (states[:, HISTORY:] - states[:, HISTORY - 1:HISTORY]) / scales
    test_categories = np.array([records[i][1] for i in test_cpu])
    test_sources = np.array([records[i][2] for i in test_cpu])
    total_force = force[..., -4:-1].norm(dim=-1).sum(-1) * 3
    stats = {
        "seed": args.seed,
        "dataset": str(args.dataset),
        "group_bones_takes": args.group_bones_takes,
        "training_config": {
            "windows_per_clip": args.windows_per_clip, "clip_limit": args.clip_limit,
            "batch": args.batch, "epochs": args.epochs, "flow_epochs": args.flow_epochs,
        },
        "counts": {key: {"windows": len(ids), "recordings": len(source_sets[key]),
                         "categories": dict(Counter(records[i][1] for i in ids.tolist()))}
                   for key, ids in splits.items()},
        "force_max_point_total_slew": budgets.tolist(),
        "force_total_N_mean": total_force.mean().item(),
        "force_total_N_p95": torch.quantile(total_force.flatten(), 0.95).item(),
        "counts_by_contacts": dict(Counter(counts.tolist())),
        "models": {},
    }
    print(json.dumps({k: v for k, v in stats.items() if k != "models"}), flush=True)
    (args.output / "windows.json").write_text(json.dumps(records))
    all_metrics = {}
    predictions = {}
    for mode, flow in (("history", False), ("causal", False), ("delayed", False),
                       ("metadata", False), ("oracle", False), ("oracle", True)):
        name = "oracle_flow" if flow else mode
        if flow and not args.flow_epochs:
            continue
        model, training = train_model(mode, flow, history, target, force, train, valid, args)
        results, forecasts = evaluate(model, mode, flow, history, target, force, test, donor,
                                      fk, scales, states, args.batch, args.seed)
        stats["models"][name] = {"training": training, **{
            case: summarize(values, test_categories) for case, values in results.items()
        }}
        if "reversed" in forecasts:
            stats["models"][name]["paired_vs_reversed"] = summarize(
                metrics(fk, forecasts["reversed"], forecasts["paired"]), test_categories
            )
            if mode != "metadata":
                stats["models"][name]["reversal_response"] = reversal_response(
                    fk, forecasts, force[test], body[test], local[test], mode, args.batch
                )
        all_metrics[name] = results
        predictions[name] = forecasts["paired"]
        if name != "history":
            stats["models"][name]["fk_improvement_ci95_cm"] = {
                case: bootstrap_improvement(base["body_fk_cm"], results["paired"]["body_fk_cm"],
                                            test_sources, args.seed)
                for case, base in {
                    "history": all_metrics["history"]["paired"], "shuffled": results["shuffled"]
                }.items()
            }
        torch.save({"model": model.state_dict(), "history_mean": center, "history_std": spread,
                    "motion_scale": scales, "condition": mode, "flow": flow, "joint_names": names},
                   args.output / f"{name}.pt")
        (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(name, {case: value["all"] for case, value in stats["models"][name].items()
                     if isinstance(value, dict) and "all" in value}, flush=True)

    np.savez_compressed(args.output / "heldout.npz",
                        states=states[test].cpu().numpy(), force=force[test].cpu().numpy(),
                        contacts=body[test].cpu().numpy(), local=local[test].cpu().numpy(),
                        test_indices=test.cpu().numpy(), donor_indices=donor.cpu().numpy(),
                        **{key: value.cpu().numpy() for key, value in predictions.items()})


if __name__ == "__main__":
    main()
