"""Train a G1 motion transformer on dense and masked surface spring prompts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dense_next import surface_points
from pilot import (
    DT, FRAMES, FUTURE, HISTORY, ROOT, STATE_DIM, SurfaceFK, bootstrap_improvement,
    inverse, load_windows, metrics, multiply, quat_apply, rotation_quat, rotation_vector,
    summarize, yaw_quat,
)
from active_adaptation.utils.body_guidance import limit_force, slew_force


GAIN = 40.0
LAYOUT_SEED = 20261123
SPLIT_SEED = 20261122
BASES = ((3, .45), (5, .45), (10, .05), (40, .05))
PRETRAIN_FRACTION = .25


def make_fk(names, device):
    geometry = json.loads((ROOT / "artifacts/ceer_segmented_jacobian/geometry.json").read_text())
    return SurfaceFK(geometry, names, device)


@torch.no_grad()
def prepare(args, device):
    windows, records, names = load_windows(
        ROOT / "dataset/compliance_mix_29h_v1", SPLIT_SEED, args.windows_per_clip, args.clip_limit, True
    )
    fk = make_fk(names, device)
    points = surface_points(LAYOUT_SEED)
    bodies = torch.tensor([fk.bodies.index(point["body"]) for point in points], device=device)
    local = torch.tensor([point["offset"] for point in points], device=device)
    states, forces = [], []
    for batch in torch.from_numpy(windows).split(256):
        batch = batch.to(device)
        p, r, q = batch[..., :3], F.normalize(batch[..., 3:7], dim=-1), batch[..., 7:]
        frame = inverse(yaw_quat(r[:, HISTORY - 1]))
        anchor = p[:, HISTORY - 1].clone()
        anchor[:, 2] = 0
        frame_r = frame[:, None].expand_as(r)
        state = torch.cat((quat_apply(frame_r, p - anchor[:, None]),
                           rotation_vector(multiply(frame_r, r)), q), -1)
        n = len(batch)
        pos, quat = fk(q.flatten(0, 1), p.flatten(0, 1), r.flatten(0, 1))
        pos = pos[:, bodies].reshape(n, FRAMES, len(points), 3)
        quat = quat[:, bodies].reshape(n, FRAMES, len(points), 4)
        xyz = pos + quat_apply(quat, local.expand_as(pos))
        raw = quat_apply(frame[:, None, None].expand(n, FRAMES - 1, len(points), 4),
                         GAIN * xyz.diff(dim=1))
        states.append(state.cpu())
        forces.append(raw.cpu())
    groups = {s: {row[2] for row in records if row[0] == s} for s in ("train", "validation", "test")}
    assert sum(map(len, groups.values())) == len(set.union(*groups.values()))
    states, forces = torch.cat(states), torch.cat(forces)
    assert states.isfinite().all() and forces.isfinite().all()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists():
        raise FileExistsError(args.cache)
    tmp = args.cache.with_suffix(".tmp")
    torch.save({"records": records, "names": names, "points": points, "state": states, "raw": forces,
                "config": {"gain": GAIN, "split_seed": SPLIT_SEED, "layout_seed": LAYOUT_SEED,
                           "clip_limit": args.clip_limit, "windows_per_clip": args.windows_per_clip}}, tmp)
    tmp.rename(args.cache)
    print("Cached", len(records), "windows; recordings:", {k: len(v) for k, v in groups.items()}, flush=True)


def sample_order(n, points, generator, device):
    return torch.rand((n, points), generator=generator, device=device).argsort(dim=-1)


def inject_force(raw, order, policy, *, active=None, verify=False):
    """Select physically active points, enforce budgets, then gate observations."""
    cols = order[:, None, :, None].expand(-1, raw.shape[1], -1, 3)
    selected = raw.gather(2, cols)
    if active is not None:
        selected = selected * active[:, None, :, None]
    limited = limit_force(selected.flatten(0, 1)).reshape_as(selected)
    rate = limited.diff(dim=1, prepend=torch.zeros_like(limited[:, :1])).norm(dim=-1)
    if bool((rate > 150 * DT + 1e-6).any()):
        value = torch.zeros_like(limited[:, 0])
        applied = []
        for desired in limited.unbind(1):
            value = slew_force(value, desired, DT)
            applied.append(value)
        limited = torch.stack(applied, dim=1)
    if verify:
        magnitude = limited.norm(dim=-1)
        slew = limited.diff(dim=1, prepend=torch.zeros_like(limited[:, :1])).norm(dim=-1) / DT
        assert magnitude.max() <= 30.001 and magnitude.sum(-1).max() <= 60.001 and slew.max() <= 150.01
    # Mask indicates observed contacts even when an observed force is zero.
    inp = torch.cat((limited, torch.ones_like(limited[..., :1])), dim=-1)
    if active is not None:
        inp = inp * active[:, None, :, None]
    if policy == "causal":
        inp[:, HISTORY:] = 0
    return inp


class MaskedMotion(nn.Module):
    def __init__(self, points, width=96):
        super().__init__()
        self.state = nn.Linear(STATE_DIM + 1, width)
        self.contact = nn.Linear(points * 4, width, bias=False)
        self.position = nn.Parameter(torch.randn(FRAMES - 1, width) * .02)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0., batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, STATE_DIM))
        self.points = points

    def forward(self, state, selected, order):
        state = F.pad(state, (0, 1, 0, FUTURE - 1))
        state[:, :HISTORY, -1] = 1
        force = selected.new_zeros(len(selected), FRAMES - 1, self.points, 4)
        idx = order[:, None, :, None].expand_as(selected)
        force.scatter_(2, idx, selected)
        values = self.state(state) + self.contact(force.flatten(2)) + self.position
        return self.output(self.blocks(values)[:, HISTORY - 1:])


@torch.no_grad()
def score(fk, forecast, actual):
    result = metrics(fk, forecast, actual)
    root = (forecast[:, :5, :3] - actual[:, :5, :3]).norm(dim=-1)
    result["root_first_100ms_cm"] = root.mean(-1) * 100
    return result


def point_paths(fk, motion, bodies, local):
    n, times = motion.shape[:2]
    flat = motion.flatten(0, 1)
    pos, quat = fk(flat[:, 6:], flat[:, :3], rotation_quat(flat[:, 3:6]))
    rows = torch.arange(len(flat), device=motion.device)[:, None]
    bodies = bodies.repeat_interleave(times, dim=0)
    local = local.repeat_interleave(times, dim=0)
    return (pos[rows, bodies] + quat_apply(quat[rows, bodies], local)).reshape(n, times, -1, 3)


@torch.no_grad()
def evaluate(model, fk, state, history, scale, raw, ids, donor, order, policy, body, local, batch, history_only):
    output = {variant: [] for variant in ("paired", "shuffled", "absent", "zero", "reversed", "noise_0p1")}
    reversals, strengths = [], []
    noise_gen = torch.Generator(device=raw.device).manual_seed(SPLIT_SEED + 37)
    for indexes, others, contact in zip(ids.split(batch), donor.split(batch), order.split(batch)):
        force = inject_force(raw[indexes], contact, policy, verify=True)
        strengths.append(force[..., :3].norm(dim=-1).sum(-1).mean(-1))
        reversed_force = torch.cat((-force[..., :3], force[..., 3:]), dim=-1)
        measured = force.clone()
        measured[..., :3] += .1 * torch.randn(measured[..., :3].shape, generator=noise_gen, device=raw.device) * force[..., 3:]
        actual = state[indexes, HISTORY:]
        start = state[indexes, HISTORY - 1:HISTORY]
        actual_points = point_paths(fk, actual[:, :5], body[contact], local[contact])
        trials = {}
        for variant, inputs in (
            ("paired", force), ("shuffled", inject_force(raw[others], contact, policy)),
            ("absent", torch.zeros_like(force)),
            ("zero", torch.cat((torch.zeros_like(force[..., :3]), force[..., 3:]), dim=-1)),
            ("reversed", reversed_force), ("noise_0p1", measured),
        ):
            if history_only:
                inputs = torch.zeros_like(inputs)
            forecast = model(history[indexes], inputs, contact) * scale + start
            paths = point_paths(fk, forecast[:, :5], body[contact], local[contact])
            values = score(fk, forecast, actual)
            values["contact_first_100ms_cm"] = (paths - actual_points).norm(dim=-1).mean((1, 2)) * 100
            output[variant].append(values)
            if variant in {"paired", "reversed"}:
                trials[variant] = paths
        prompt = force[:, HISTORY - 1:HISTORY, :, :3] if policy == "causal" else force[:, HISTORY - 1:HISTORY + 4, :, :3]
        prompt = prompt.expand(-1, 5, -1, -1)
        mass = prompt.norm(dim=-1).sum((1, 2))
        projected = ((trials["paired"] - trials["reversed"]) * prompt).sum((1, 2, 3))
        reversals.append(torch.stack((projected * 100 / mass.clamp_min(1e-6), mass), dim=-1))
    values = {
        variant: {key: torch.cat([v[key] for v in chunks]) for key in chunks[0]}
        for variant, chunks in output.items()
    }
    return values, torch.cat(reversals), torch.cat(strengths)


def get_donors(records, ids, device):
    rng = np.random.default_rng(SPLIT_SEED + 98)
    categories = {}
    for i in ids:
        categories.setdefault(records[i][1], []).append(i)
    output = []
    for i in ids:
        choices = [j for j in categories[records[i][1]] if records[j][2] != records[i][2]]
        output.append(rng.choice(choices))
    return torch.as_tensor(output, device=device)


def fixed_orders(count, points, device, offset):
    indices = sample_order(count, points, torch.Generator(device="cpu").manual_seed(SPLIT_SEED + offset), "cpu")
    return indices.to(device)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def train(args, device):
    data = torch.load(args.cache, map_location="cpu", weights_only=True)
    records, names, points = (data[k] for k in ("records", "names", "points"))
    state, raw = (data[k].to(device) for k in ("state", "raw"))
    ids_cpu = {s: [i for i, row in enumerate(records) if row[0] == s]
               for s in ("train", "validation", "test")}
    train_ids, validation, test = (torch.as_tensor(ids_cpu[k], device=device) for k in ids_cpu)
    args.output.mkdir(parents=True, exist_ok=False)
    fk = make_fk(names, device)
    body = torch.tensor([fk.bodies.index(x["body"]) for x in points], device=device)
    local = torch.tensor([x["offset"] for x in points], device=device)
    scale = (state[train_ids, HISTORY:] - state[train_ids, HISTORY - 1:HISTORY]).std((0, 1)).clamp_min(.015)
    center = state[train_ids, :HISTORY].mean((0, 1))
    spread = state[train_ids, :HISTORY].std((0, 1)).clamp_min(.025)
    history = (state[:, :HISTORY] - center) / spread
    target = (state[:, HISTORY:] - state[:, HISTORY - 1:HISTORY]) / scale
    npoints = len(points)
    fixed_train = fixed_orders(len(state), npoints, device, args.seed)
    fixed_counts = torch.arange(len(state), device=device) % 2 * 2 + 3
    slot_ids = torch.arange(5, device=device)
    validation_order = fixed_orders(len(validation), npoints, device, 331)
    test_order = fixed_orders(len(test), npoints, device, 443)
    val_force = {k: inject_force(raw[validation], validation_order[:, :k], args.policy, verify=True)
                 for k in (3, 5)}
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 27)
    rng = np.random.default_rng(args.seed + 5)
    model = MaskedMotion(npoints).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=8e-5)
    dense_epochs = max(1, int(args.epochs * PRETRAIN_FRACTION))
    config = dict(mode=args.mode, policy=args.policy, seed=args.seed, epochs=args.epochs, batch=args.batch,
                  dense_epochs=dense_epochs if args.mode == "pretrain" else 0,
                  counts={key: len(vals) for key, vals in ids_cpu.items()}, cache=data["config"])
    save_json(args.output / "config.json", config)
    print(config, flush=True)
    best = float("inf")
    best_epoch = 0
    progress = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        perm = torch.randperm(len(train_ids), generator=generator, device=device)
        for indexes in train_ids[perm].split(args.batch):
            if args.mode == "direct":
                order = fixed_train[indexes, :5]
                force = inject_force(raw[indexes], order, args.policy,
                                     active=slot_ids < fixed_counts[indexes, None])
            elif args.mode == "history":
                order = fixed_train[indexes, :3]
                force = raw.new_zeros(len(indexes), FRAMES - 1, 3, 4)
            else:
                n = npoints if args.mode == "pretrain" and epoch < dense_epochs else int(
                    rng.choice([k for k, _ in BASES], p=[p for _, p in BASES])
                )
                order = sample_order(len(indexes), npoints, generator, device)[:, :n]
                force = inject_force(raw[indexes], order, args.policy)
            pred = model(history[indexes], force, order)
            loss = F.mse_loss(pred, target[indexes])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach())
        sched.step()
        model.eval()
        with torch.no_grad():
            val_scores = []
            for k, cond in val_force.items():
                if args.mode == "history":
                    cond = torch.zeros_like(cond)
                for hist, f, order, expected in zip(history[validation].split(args.batch), cond.split(args.batch),
                          validation_order[:, :k].split(args.batch), target[validation].split(args.batch)):
                    val_scores.append((model(hist, f, order) - expected).square().mean((1, 2)))
            val = torch.cat(val_scores).mean().item()
        if epoch >= min(dense_epochs, args.epochs - 1) and val < best:
            best = val
            best_epoch = epoch + 1
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "history_mean": center.cpu(), "history_std": spread.cpu(), "motion_scale": scale.cpu(),
                        "points": points, "joints": names, "config": config}, args.output / "best.pt")
        progress.append({"epoch": epoch + 1, "loss": torch.stack(losses).mean().item(), "validation": val})
        save_json(args.output / "progress.json", progress)
        print(progress[-1], flush=True)
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True)["model"])
    categories = np.array([records[i][1] for i in ids_cpu["test"]])
    sources = np.array([records[i][2] for i in ids_cpu["test"]])
    donor = get_donors(records, ids_cpu["test"], device)
    summary = {"config": config, "validation": {"epoch": best_epoch, "scaled_mse": best}, "test": {}}
    detailed = {"categories": categories, "recordings": sources, "test_ids": test.cpu().numpy()}
    for k in (3, 5):
        values, reverse, strengths = evaluate(model, fk, state, history, scale, raw, test, donor,
            test_order[:, :k], args.policy, body, local, args.batch, args.mode == "history")
        keep = reverse[:, 1] > 1e-4
        summary["test"][str(k)] = {
            "conditions": {case: summarize(v, categories) for case, v in values.items()},
            "force_total_N_mean_including_masked_frames": strengths.mean().item(),
            "reversal_projected_cm": reverse[keep, 0].mean().item(),
            "reversal_fraction_positive": (reverse[keep, 0] > 0).float().mean().item(),
            "paired_vs_shuffled_fk_improvement_ci95_cm": bootstrap_improvement(
                values["shuffled"]["body_fk_cm"], values["paired"]["body_fk_cm"], sources, args.seed),
        }
        for case, results in values.items():
            for metric, value in results.items():
                detailed[f"{k}__{case}__{metric}"] = value.cpu().numpy()
        detailed[f"{k}__projected_reversal"] = reverse.cpu().numpy()
    np.savez_compressed(args.output / "per_window.npz", **detailed)
    save_json(args.output / "summary.json", summary)
    print("complete", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-limit", type=int, default=900)
    parser.add_argument("--windows-per-clip", type=int, default=2)
    parser.add_argument("--mode", choices=("direct", "masked", "pretrain", "history"))
    parser.add_argument("--policy", choices=("whole", "causal"))
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.action == "prepare":
        prepare(args, device)
    else:
        if args.output is None or args.mode is None or args.policy is None:
            parser.error("Training requires --output, --mode and --policy")
        train(args, device)


if __name__ == "__main__":
    main()
