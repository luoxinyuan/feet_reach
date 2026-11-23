"""Compare full-body virtual force prompts for one-step G1 motion decoding."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pilot import DT, ROOT, load_windows, inverse, rotation_quat, rotation_vector, yaw_quat
from active_adaptation.utils.body_guidance import limit_force, slew_force
from active_adaptation.utils.segmented_guidance import geometry_jacobians
from active_adaptation.utils.surface_fk import SurfaceFK, multiply
from active_adaptation.utils.surface_guidance import quat_apply


POINT_CAP = 30.0
TOTAL_CAP = 60.0
SLEW_CAP = 150.0
CALIBRATION_P90 = 20.0
NOISE_LEVELS = (0.0, 0.1, 0.5)
METHODS = ("spring80", "spring_matched", "wrench_ridge", "balanced_cartesian")


def surface_points(seed):
    path = ROOT / "artifacts/surface_compliance/geometry"
    manifest = json.loads((path / "manifest.json").read_text())
    triangles = np.load(path / "triangles.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    parts_by_body = {}
    for part in manifest["parts"]:
        parts_by_body.setdefault(part["body"], []).append(part)
    points = []
    for body, parts in parts_by_body.items():
        mesh = np.concatenate([triangles[p["start"]:p["end"]] for p in parts])
        edges_a, edges_b = mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]
        area = np.linalg.norm(np.cross(edges_a, edges_b), axis=-1)
        subset = mesh[rng.choice(len(mesh), size=64, p=area / area.sum())]
        bary = rng.dirichlet((1, 1, 1), len(subset))
        offsets = np.einsum("tvc,tv->tc", subset, bary)
        first, second = np.unravel_index(
            np.square(offsets[:, None] - offsets).sum(-1).argmax(), (len(subset), len(subset))
        )
        points.extend({"body": body, "offset": offsets[i].tolist()} for i in (first, second))
    return points


@torch.no_grad()
def geometry_batches(windows, geometry, names, points, device, sample):
    """Build force bases and next-state labels in the current heading frame."""
    step = torch.as_tensor(sample, device=device)
    memories, targets, poses, bases, maps = [], [], [], [], []
    for raw_np in np.array_split(windows, (len(windows) + 31) // 32):
        n, frames, _ = raw_np.shape
        flat = raw_np.reshape(-1, 36)
        pos, jac = geometry_jacobians(flat[:, 7:], flat[:, :3], flat[:, 3:7], names, points, geometry)
        pos = torch.as_tensor(pos, dtype=torch.float32, device=device).reshape(n, frames, -1, 3)
        jac = torch.as_tensor(jac, dtype=torch.float32, device=device).reshape(n, frames, -1, 3, 35)[:, :-1]
        raw = torch.as_tensor(raw_np, device=device)
        p, r, q = raw[..., :3], F.normalize(raw[..., 3:7], dim=-1), raw[..., 7:]
        heading = yaw_quat(r[:, :-1])
        frame = inverse(heading)
        relative = multiply(frame, r[:, :-1])
        delta_rotation = rotation_vector(multiply(r[:, 1:], inverse(r[:, :-1])))
        delta = torch.cat((p.diff(dim=1), delta_rotation, q.diff(dim=1)), -1)
        next_state = torch.cat((quat_apply(frame, delta[..., :3]), quat_apply(frame, delta[..., 3:6]), delta[..., 6:]), -1)
        previous = torch.cat((torch.zeros_like(next_state[:, :1]), delta[:, :-1]), dim=1)
        previous = torch.cat((quat_apply(frame, previous[..., :3]),
                              quat_apply(frame, previous[..., 3:6]), previous[..., 6:]), -1)
        state = torch.cat((p[:, :-1, 2:3], rotation_vector(relative), q[:, :-1], previous), -1)
        points_delta = pos.diff(dim=1)
        # Characteristic increments for root translations, root angles and joints.
        # S normalizes the target; translating 1 cm and rotating 0.025 rad
        # have equal normalized importance. All three root rotations enter the fit.
        scales = delta.new_tensor([0.01] * 3 + [0.025] * 32)
        j = jac.flatten(2, 3)
        b = j * scales / 0.01
        gram = b.transpose(-1, -2) @ b
        eye = torch.eye(35, device=device)
        d = (delta / scales).unsqueeze(-1)
        # J^T f fits the full-body normalized increment, with ridge stabilizing
        # poorly observable modes. Unit gain is calibrated later on training only.
        wrench = (b @ torch.linalg.solve(gram + 0.15 * eye, d)).squeeze(-1).reshape_as(points_delta)
        # SVD-equivalent regularized whitening of the forward Cartesian
        # encoding. This sits between plain springs and the inverse allocation.
        eig, axes = torch.linalg.eigh(gram)
        coeff = (axes.transpose(-1, -2) @ d) * (eig.clamp_min(0) + 0.15).rsqrt()[..., None]
        balanced = (b @ (axes @ coeff)).squeeze(-1).reshape_as(points_delta)
        memories.append(state[:, step])
        targets.append(next_state[:, step])
        poses.append(torch.cat((p[:, :-1], relative, q[:, :-1]), -1)[:, step])
        bases.append(torch.stack((80 * points_delta, wrench, balanced), 1))
        maps.append(frame)
    return tuple(torch.cat(values) for values in (memories, targets, poses, bases, maps))


@torch.no_grad()
def make_force(raw, heading, train_windows, steps, extra_spring_gains):
    force_by_method, stats = {}, {}
    gain_p90 = torch.quantile(raw[train_windows].norm(dim=-1).sum(-1).transpose(0, 1).flatten(1), 0.90, dim=1)
    gains = torch.cat((gain_p90.new_ones(1), CALIBRATION_P90 / gain_p90))
    variants = list(zip(METHODS, (0, 0, 1, 2), gains))
    variants.extend((f"spring{k:g}", 0, raw.new_tensor(k / 80)) for k in extra_spring_gains)
    for method, src, gain in variants:
        desired = raw[:, src] * gain
        limited = limit_force(desired.flatten(0, 1), POINT_CAP, TOTAL_CAP).reshape_as(desired)
        current = torch.zeros_like(desired[:, 0])
        applied = []
        for target in limited.unbind(1):
            current = slew_force(current, target, DT, SLEW_CAP)
            applied.append(current)
        applied = torch.stack(applied, 1)
        norm = applied.norm(dim=-1)
        rate = torch.diff(applied, dim=1, prepend=torch.zeros_like(applied[:, :1])).norm(dim=-1) / DT
        total = norm.sum(-1)
        assert total.amax() <= TOTAL_CAP + 1e-3
        assert norm.amax() <= POINT_CAP + 1e-3
        assert rate.amax() <= SLEW_CAP + 1e-2
        force_by_method[method] = quat_apply(heading[:, :, None].expand(*applied.shape[:-1], 4), applied)[:, steps]
        stats[method] = {
            "gain_multiplier": gain.item(),
            "point_max_N": norm.amax().item(), "total_max_N": total.amax().item(), "slew_max_N_s": rate.amax().item(),
            "total_mean_N": total.mean().item(), "total_p90_N": total.quantile(.90).item(),
            "fraction_budget_limited": (limited - desired).abs().amax((2, 3)).gt(1e-4).float().mean().item(),
            "fraction_slew_limited": (limited - applied).abs().amax((2, 3)).gt(1e-4).float().mean().item(),
            "active_points_mean": (norm > .05).float().sum(-1).mean().item(),
            "fraction_force_against_point_motion": (
                ((applied * raw[:, 0]).sum(-1) < -1e-5) & (norm > .05)
            ).sum().item() / max((norm > .05).sum().item(), 1),
        }
    return force_by_method, stats


class Decoder(nn.Module):
    def __init__(self, state_dim, force_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + force_dim, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 35),
        )

    def forward(self, state, force):
        return self.model(torch.cat((state, force.flatten(1)), -1))


@torch.no_grad()
def measure(fk, pose, target, predicted):
    def apply_delta(d):
        r = multiply(rotation_quat(d[:, 3:6]), pose[:, 3:7])
        p = pose[:, :3] + d[:, :3]
        q = pose[:, 7:] + d[:, 6:]
        return fk(q, p, r)[0]

    error = predicted - target
    fk_error = (apply_delta(predicted) - apply_delta(target)).norm(dim=-1)
    residual = (error[:, 6:]).abs().mean(-1)
    wrists = [fk.bodies.index(f"{side}_wrist_yaw_link") for side in ("left", "right")]
    feet = [fk.bodies.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    return torch.stack((
        fk_error.mean(-1) * 1000, error[:, :3].norm(dim=-1) * 1000,
        error[:, 3:6].norm(dim=-1) * (180 / torch.pi),
        residual * (180 / torch.pi),
        fk_error[:, wrists].mean(-1) * 1000, fk_error[:, feet].mean(-1) * 1000,
    ), -1)


METRICS = ("mean_fk_mm", "root_translation_mm", "root_rotation_deg", "joint_mae_deg", "wrist_fk_mm", "foot_fk_mm")


def noise_like(x, seed):
    return torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(seed))


def fit(state, force, target, train, valid, fk, poses, args, name):
    torch.manual_seed(args.seed + 1)
    scale_s = state[train].std(dim=0).clamp_min(0.002)
    center_s = state[train].mean(dim=0)
    s = (state - center_s) / scale_s
    residual = target - state[:, -35:]
    scale_y = residual[train].std(dim=0).clamp_min(0.0001)
    force_scale = force[train].square().mean().sqrt().clamp_min(0.1)
    model = Decoder(state.shape[-1], force.shape[1] * 3).to(force.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=0.0001)
    val_noise = noise_like(force[valid], args.seed + 3) * .1
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for ids in train[torch.randperm(len(train), device=train.device)].split(384):
            f = force[ids]
            if name != "history":
                f = f + .1 * torch.randn_like(f)
            y = model(s[ids], f / force_scale)
            loss = F.mse_loss(y, residual[ids] / scale_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            cond = force[valid] if name == "history" else force[valid] + val_noise
            y = model(s[valid], cond / force_scale) * scale_y + state[valid, -35:]
            score = measure(fk, poses[valid], target[valid], y)[:, 0].mean().item()
        if score < best:
            best = score
            weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if epoch == 0 or (epoch + 1) % 6 == 0 or epoch + 1 == args.epochs:
            print(name, epoch + 1, "val FK mm", round(score, 4), flush=True)
    model.load_state_dict(weights)
    return model, s, scale_y, force_scale, {"validation_fk_mm_noise_0p1": best, "epoch": best_epoch,
        "state_mean": center_s.cpu(), "state_scale": scale_s.cpu()}


def donors(rows, ids, seed):
    rng = np.random.default_rng(seed)
    windows = {}
    for index in ids:
        windows.setdefault(rows[index][1], []).append(index)
    return [rng.choice([j for j in windows[rows[i][1]] if rows[j][2] != rows[i][2]]) for i in ids]


def summary(values, categories):
    def aggregate(rows):
        return dict(zip(METRICS, np.mean(rows, axis=0).tolist()))
    return {
        "all": aggregate(values),
        "by_category": {cat: aggregate(values[categories == cat]) for cat in sorted(set(categories))},
    }


def paired_ci(best, others, recordings, seed):
    _, index = np.unique(recordings, return_inverse=True)
    count = np.bincount(index)
    ids = np.random.default_rng(seed).integers(len(count), size=(2000, len(count)))
    dif = np.bincount(index, weights=others - best)
    return np.quantile(dif[ids].sum(1) / count[ids].sum(1), [0.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-limit", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--extra-spring-gains", type=float, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(set(args.extra_spring_gains)) != len(args.extra_spring_gains) or any(
        k <= 0 or k == 80 or not np.isfinite(k) for k in args.extra_spring_gains
    ):
        parser.error("Extra spring gains must be distinct finite positives other than 80")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    windows, rows, names = load_windows(ROOT / "dataset/compliance_mix_29h_v1", args.seed, 1, args.clip_limit, True)
    geometry = json.loads((ROOT / "artifacts/ceer_segmented_jacobian/geometry.json").read_text())
    layout_seed = args.seed if args.layout_seed is None else args.layout_seed
    points = surface_points(layout_seed)
    fk = SurfaceFK(geometry, names, device)
    steps = list(range(6, windows.shape[1] - 1, 4))
    inputs, target, poses, raw, heading = geometry_batches(windows, geometry, names, points, device, steps)
    frame_count = len(steps)
    splits = {tag: [i for i, row in enumerate(rows) if row[0] == tag] for tag in ("train", "validation", "test")}
    split_groups = {tag: {rows[i][2] for i in ids} for tag, ids in splits.items()}
    assert not any(split_groups[a] & split_groups[b] for a, b in (
        ("train", "test"), ("train", "validation"), ("test", "validation")
    ))
    forces, force_stats = make_force(raw, heading, splits["train"], steps, args.extra_spring_gains)
    del raw, heading, windows
    inputs, target, poses = (tensor.flatten(0, 1) for tensor in (inputs, target, poses))
    assert inputs.isfinite().all() and target.isfinite().all()

    def flatten_ids(ids):
        return (torch.tensor(ids, device=device)[:, None] * frame_count + torch.arange(frame_count, device=device)).flatten()

    train, valid, test = (flatten_ids(splits[tag]) for tag in ("train", "validation", "test"))
    shuffled = flatten_ids(donors(rows, splits["test"], args.seed))
    categories = np.repeat([rows[i][1] for i in splits["test"]], frame_count)
    recordings = np.repeat([rows[i][2] for i in splits["test"]], frame_count)
    stats = {
        "config": {"seed": args.seed, "epochs": args.epochs, "clip_limit": args.clip_limit,
                   "layout_seed": layout_seed, "extra_spring_gains_N_m": args.extra_spring_gains,
                   "frame_offsets": steps, "calibration_train_raw_p90_N": CALIBRATION_P90,
                   "decoder_training_noise_sigma_N": 0.1, "selection_noise_sigma_N": 0.1,
                   "force_limits_point_total_slew": [POINT_CAP, TOTAL_CAP, SLEW_CAP]},
        "splits": {tag: {"windows": len(idx), "transitions": len(idx)*frame_count,
                         "recordings": len(split_groups[tag]), "categories": dict(Counter(rows[i][1] for i in idx))}
                   for tag, idx in splits.items()},
        "points": points, "force": force_stats, "decoders": {},
    }
    (args.output / "windows.json").write_text(json.dumps(rows))
    print(json.dumps({key: stats[key] for key in ("splits", "force")}), flush=True)
    errors = {}
    with torch.no_grad():
        for name, y in (("hold", torch.zeros_like(target[test])), ("constant_velocity", inputs[test, -35:])):
            result = measure(fk, poses[test], target[test], y).cpu().numpy()
            stats["decoders"][name] = {"noise_0": summary(result, categories)}
    for name in ("history", *forces):
        force = (torch.zeros_like(forces[METHODS[0]]) if name == "history" else forces[name]).flatten(0, 1)
        assert force.isfinite().all()
        model, state, sy, sf, fit_stats = fit(inputs, force, target, train, valid, fk, poses, args, name)
        stats["decoders"][name] = {k: v for k, v in fit_stats.items() if not isinstance(v, torch.Tensor)}
        with torch.no_grad():
            noise = noise_like(force[test], args.seed + 5)
            cases = {f"noise_{sigma:g}": force[test] + sigma * noise for sigma in NOISE_LEVELS}
            cases["shuffled_noise_0.1"] = force[shuffled] + .1 * noise
            if name == "history":
                cases = {"noise_0": force[test]}
            for key, f in cases.items():
                y = model(state[test], f / sf) * sy + inputs[test, -35:]
                result = measure(fk, poses[test], target[test], y).cpu().numpy()
                errors[name, key] = result
                stats["decoders"][name][key] = summary(result, categories)
        torch.save({"model": model.cpu().state_dict(), "residual_from_constant_velocity": True,
                    "target_scale": sy.cpu(), "force_scale": sf.cpu(),
                    "state_mean": fit_stats["state_mean"], "state_scale": fit_stats["state_scale"],
                    "points": points, "joints": names}, args.output / f"{name}.pt")
        print(name, {k: v["all"] for k, v in stats["decoders"][name].items() if isinstance(v, dict)}, flush=True)
        (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    winner = min(forces, key=lambda method: stats["decoders"][method]["validation_fk_mm_noise_0p1"])
    stats["selected_method"] = winner
    stats["selection_metric"] = "validation mean FK (mm), force component noise sigma=0.1 N"
    stats["test_improvement_over_other_ci95_mm_noise_0p1"] = {
        other: paired_ci(errors[winner, "noise_0.1"][:, 0], errors[other, case][:, 0], recordings, args.seed)
        for other, case in [("history", "noise_0"), *[(name, "noise_0.1") for name in forces if name != winner]]
    }
    (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez_compressed(args.output / "test_errors.npz", category=categories, recording=recordings,
                        **{f"{x}__{y}": z for (x, y), z in errors.items()})
    print("selected", winner, stats["test_improvement_over_other_ci95_mm_noise_0p1"], flush=True)


if __name__ == "__main__":
    main()
