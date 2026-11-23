"""Learn a motion-conditioned force refinement on searched 8KP subsets."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from scripts.force_mapping_eval.evaluate_v2 import GROUPS, fit, metrics


def prepare(root, output, horizon=5):
    with np.load(root / "data.npz") as d:
        clip, step = d["clip"], d["step"]
    windows = []
    for shard in range(6):
        with np.load(root / "fixed8_jacobian" / f"shard{shard}.npz") as d:
            windows.extend(d["windows"])
    windows = {(int(row[0]), int(row[1])): (int(row[2]), row[4:][row[4:] >= 0])
               for row in windows}
    xyz, masks, caps, blocks = [], [], [], []
    offset = 0
    with np.load(root / "masked_spring" / "geometry.npz") as d:
        for c, n in zip(*np.unique(clip, return_counts=True)):
            p, r = d[f"p{c}"], d[f"r{c}"]
            desired_at = (np.arange(n) - 1).clip(0)
            target_at = (desired_at + horizon).clip(max=len(p)-1)
            delta = (p[target_at] - p[desired_at]) @ r[:n]
            enabled = np.zeros((n, 8), dtype=bool)
            envelope = np.ones(n, np.float32)
            for start in range(0, n, 40):
                end, ids = windows[int(c), start]
                enabled[start + 1:max(start + 1, end - 2), ids] = True
                blocks.append((offset + start, offset + end, len(ids)))
                if start + 1 < end - 2:
                    envelope[start + 1] = .6
                    envelope[end - 3] = min(envelope[end - 3], .6)
            xyz.append(delta)
            masks.append(enabled)
            caps.append(envelope)
            offset += n
    assert len(np.concatenate(xyz)) == len(step)
    np.savez_compressed(input_path(output, horizon), displacement=np.concatenate(xyz),
                        enabled=np.concatenate(masks), envelope=np.concatenate(caps), blocks=blocks)


def input_path(root, horizon):
    return root / ("inputs.npz" if horizon == 5 else f"inputs_h{horizon}.npz")


def aligned_slew(force, clip, cap):
    """Reduce only magnitudes to meet the observed-frame 3 N step budget."""
    for i in range(1, len(force)):
        if clip[i] != clip[i - 1]:
            continue
        norm = np.linalg.norm(force[i], axis=-1)
        if not norm.any():
            continue
        direction = force[i] / np.maximum(norm[:, None], 1e-8)
        # Compare both forces in a common world frame before calling this function.
        projected = (force[i - 1] * direction).sum(-1)
        remainder = np.square(force[i - 1]).sum(-1) - projected**2
        limit = projected + np.sqrt(np.maximum(cap**2 - remainder, 0.))
        force[i] *= np.minimum(1., limit.clip(0.) / np.maximum(norm, 1e-8))[:, None]
    return force


def readout(width, hidden, weights, device):
    net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                        nn.SiLU(), nn.Linear(hidden, 41)).to(device)
    net.load_state_dict(weights)
    return net.eval().requires_grad_(False)


class Refiner(nn.Module):
    def __init__(self, width, angle, max_force, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 32))
        self.angle = np.tan(np.deg2rad(angle))
        self.max_force = max_force

    def forward(self, motion, displacement, enabled, envelope):
        return self.force_from_parameters(self.net(motion), displacement, enabled, envelope)

    def force_from_parameters(self, parameters, displacement, enabled, envelope):
        pred = parameters.reshape(-1, 8, 4)
        norm = displacement.norm(dim=-1, keepdim=True)
        direction = displacement / norm.clamp_min(1e-8)
        if self.angle:
            lateral = pred[..., 1:].tanh()
            lateral = lateral - (lateral * direction).sum(-1, keepdim=True) * direction
            lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp_min(1.)
            direction = nn.functional.normalize(direction + self.angle * lateral, dim=-1)
        maximum = (160. * norm).clamp_max(self.max_force)
        magnitude = maximum * pred[..., :1].sigmoid() * envelope[:, None, None]
        return magnitude * direction * enabled[..., None] * (norm > 1e-5)


def refine_parameters(generator, reader, initial, delta, mask, envelope, pose, motion,
                      weights, prior, steps, batch_size=2048):
    refined, params = [], []
    for chunk in torch.arange(len(pose), device=pose.device).split(batch_size):
        parameters = nn.Parameter(initial[chunk].clone())
        opt = torch.optim.Adam([parameters], lr=.045)
        for _ in range(steps):
            force = generator.force_from_parameters(parameters, delta[chunk], mask[chunk], envelope[chunk])
            prediction = reader(torch.cat((pose[chunk], force.flatten(1) * 2.), 1))
            loss = ((prediction - motion[chunk]) * weights).square().mean()
            loss = loss + .003 * (force - prior[chunk]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            refined.append(generator.force_from_parameters(parameters, delta[chunk],
                                                           mask[chunk], envelope[chunk]))
            params.append(parameters.detach())
    return torch.cat(refined), torch.cat(params)


@torch.no_grad()
def select_windows(blocks, mask, force, state, target, weights, reader, tail, compact=False,
                   reset=None, length=40, probe_steps=0, amplitude_cap=1.5,
                   retain_common=False, partition=None):
    """Rank all equally sized subsets by conditional motion prediction error."""
    result = mask.clone()
    ranked_scores = []
    ranked_supports = []
    # Each row belongs to one fixed, non-overlapping contact window.
    for count in (1, 2, 3):
        ids = list(combinations(range(8), count))
        supports = torch.zeros((len(ids), 8), dtype=torch.bool, device=state.device)
        supports.scatter_(1, torch.as_tensor(ids, device=state.device), True)
        group = blocks if partition is not None else blocks[blocks[:, 2] == count]
        score_parts, support_parts = [], []
        for block in group.split(48):
            if not len(block):
                continue
            start, end, _ = block.T
            offsets = torch.arange(0 if compact else 1, length, max(1, length // 10),
                                   device=state.device)
            probes = torch.minimum(start[:, None] + offsets, end[:, None] - 1)
            observed = force[probes, None] * supports[None, None, :, :, None]
            poses = state[probes, None].expand(-1, -1, len(ids), -1)
            if probe_steps:
                mag = observed.norm(dim=-1, keepdim=True)
                unit = observed / mag.clamp_min(1e-8)
                with torch.enable_grad():
                    amplitude = nn.Parameter(mag.clone())
                    opt = torch.optim.Adam([amplitude], lr=.05)
                    for _ in range(probe_steps):
                        candidate = unit * amplitude
                        feature = torch.cat((poses, candidate.flatten(-2) * 2.), -1)
                        delta = reader(feature) - target[probes, None]
                        loss = (delta * weights).square().mean() + .0001 * (amplitude - mag).square().mean()
                        opt.zero_grad(set_to_none=True)
                        amplitude.grad, = torch.autograd.grad(loss, amplitude)
                        opt.step()
                        with torch.no_grad():
                            amplitude.clamp_(0., amplitude_cap)
                observed = unit * amplitude.detach()
            feature = torch.cat((poses, observed.flatten(-2) * 2.), -1)
            delta = reader(feature) - target[probes, None]
            score = (delta * weights).square().mean((1, 3))
            chosen = supports[score.argmin(-1)]
            if partition is not None:
                score_parts.append(score.min(-1).values)
                support_parts.append(chosen)
                continue
            frames = start[:, None] + torch.arange(length, device=state.device)
            valid = (frames >= start[:, None] + (0 if compact else 1))
            valid &= frames < end[:, None] - tail
            result[frames[valid]] = chosen[:, None].expand(-1, length, -1)[valid]
        if partition is not None:
            ranked_scores.append(torch.cat(score_parts))
            ranked_supports.append(torch.cat(support_parts))
    if partition is not None:
        cost = torch.stack(ranked_scores, 1).cpu().numpy()
        budget = blocks[:, 2].cpu().numpy()
        for split_id in (0, 1, 2):
            group = np.flatnonzero(partition == split_id)
            slots = budget[group] - 1
            row, col = linear_sum_assignment(cost[group][:, slots])
            budget[group[row]] = slots[col] + 1
        assigned = torch.as_tensor(budget, device=state.device)
        blocks[:, 2] = assigned
        chosen = torch.stack(ranked_supports, 1)[torch.arange(len(blocks), device=state.device),
                                                assigned - 1]
        start, end, _ = blocks.T
        frames = start[:, None] + torch.arange(length, device=state.device)
        valid = (frames >= start[:, None]) & (frames < end[:, None] - tail)
        result[frames[valid]] = chosen[:, None].expand(-1, length, -1)[valid]
    if compact:
        changed = (result[1:] != result[:-1]).any(-1)
        retained = (result[1:] & result[:-1])[changed] if retain_common else False
        result[1:][changed] = retained
        result[reset] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--angle", type=int, default=60)
    parser.add_argument("--smooth", type=float, default=.01)
    parser.add_argument("--max-force", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--short-release", action="store_true")
    parser.add_argument("--inner-steps", type=int, default=0)
    parser.add_argument("--group-scale", action="store_true")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--subwindow", type=int, default=40)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--refit-rounds", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--count-probs", type=float, nargs=3)
    parser.add_argument("--adaptive-counts", action="store_true")
    parser.add_argument("--calibrated-counts", action="store_true")
    parser.add_argument("--retain-common", action="store_true")
    parser.add_argument("--marginal-counts", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--export-solver", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert 0 <= args.angle < 90 and 0 < args.max_force <= (3. if args.angle == 0 else 1.5)
    if args.refit_rounds and not args.inner_steps:
        parser.error("--refit-rounds requires --inner-steps")
    if args.calibrated_counts and not args.adaptive_counts:
        parser.error("--calibrated-counts requires --adaptive-counts")
    if args.retain_common and not args.compact:
        parser.error("--retain-common requires --compact")
    if args.marginal_counts and not (args.adaptive_counts and args.compact and args.search):
        parser.error("--marginal-counts requires compact adaptive search")
    if not 2 <= args.subwindow <= 40 or (args.subwindow != 40 and not args.compact):
        parser.error("--subwindow must be 2–40 and requires --compact when changed")
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.export_solver and not (
        args.test and args.angle == 0 and args.horizon == 2 and args.calibrated_counts
        and args.compact and args.search and args.inner_steps and not args.refit_rounds
        and not args.marginal_counts and args.max_force == 1.5
    ):
        parser.error("--export-solver is for the calibrated two-frame aligned solver")
    base = args.input
    root = base / "fixed8_learned_force"
    root.mkdir(exist_ok=True)
    if args.prepare:
        prepare(base, root, args.horizon)
        return
    method = f"angle{args.angle}_smooth{args.smooth}" + ("_search" if args.search else "")
    method += ("_short" if args.short_release else "")
    method += (f"_inner{args.inner_steps}" if args.inner_steps else "")
    method += ("_groups" if args.group_scale else "")
    method += (f"_width{args.hidden}" if args.hidden != 256 else "")
    method += (f"_h{args.horizon}" if args.horizon != 5 else "")
    method += (f"_block{args.subwindow}" if args.subwindow != 40 else "")
    method += ("_compact" if args.compact else "")
    method += (f"_refit{args.refit_rounds}" if args.refit_rounds else "")
    method += (f"_cap{args.max_force}" if args.max_force != 1.5 else "")
    method += (f"_probes{args.probe_steps}" if args.probe_steps else "")
    method += ("_retain" if args.retain_common else "")
    method += ("_marginal" if args.marginal_counts else "")
    if args.count_probs:
        if any(p <= 0 for p in args.count_probs) or not np.isclose(sum(args.count_probs), 1.):
            parser.error("--count-probs must be positive and sum to 1")
        method += "_counts" + "-".join(str(p) for p in args.count_probs)
    if args.adaptive_counts:
        method += "_adaptive"
        if args.calibrated_counts:
            method += "_calibrated"
    path = root / method / f"seed{args.seed}"
    path.mkdir(parents=True, exist_ok=True)
    with np.load(base / "data.npz") as d:
        clip, step, split, state = (d[k] for k in ("clip", "step", "split", "state"))
        y = np.c_[d["delta_q"], d["delta_root_1"], d["delta_root_5"]]
        width = d["greedy3"].shape[1]
    with np.load(input_path(root, args.horizon)) as d:
        displacement, enabled, envelope, blocks = (d[k] for k in
                                                     ("displacement", "enabled", "envelope", "blocks"))
    if args.count_probs or args.adaptive_counts:
        rng = np.random.default_rng(643221)
        budgets = blocks[:, 2].copy()
        if args.count_probs:
            budgets = rng.choice((1, 2, 3), size=len(blocks), p=args.count_probs)
        if args.adaptive_counts:
            # Keep the aggregate count budget, allocating larger supports to
            # windows with higher normalized motion energy.
            scale_by_group = np.concatenate([
                np.full(sl.stop - sl.start,
                        max(float(np.sqrt(np.square(y[split == 0, sl]).sum(1).mean())), .0005))
                for sl in GROUPS.values()
            ])
            energy = np.linalg.norm(y / scale_by_group, axis=-1)
            activity = np.asarray([np.square(energy[a:b]).mean() for a, b, _ in blocks])
            if args.calibrated_counts:
                train_activity = activity[split[blocks[:, 0]] == 0]
                thresholds = np.quantile(train_activity,
                                         np.cumsum(args.count_probs or (.4, .3, .3))[:2])
                budgets = np.searchsorted(thresholds, activity) + 1
            else:
                for partition in (0, 1, 2):
                    subset = np.flatnonzero(split[blocks[:, 0]] == partition)
                    order = subset[np.argsort(activity[subset], kind="stable")]
                    budgets[order] = np.sort(budgets[subset])
        for block, k in zip(blocks, budgets):
            start, end, _ = block
            energy = np.linalg.norm(displacement[start:end], axis=-1).mean(0)
            priority = enabled[start:end].any(0)
            ids = np.lexsort((-energy, -priority.astype(int)))[:k]
            enabled[start:end] = False
            enabled[start + 1:max(start + 1, end - 2), ids] = True
            block[2] = k
    tail = 0 if args.short_release or args.compact else 2
    if args.compact:
        for start, end, _ in blocks:
            if end - start > 3:
                enabled[start:end] = enabled[start + 1].copy()
        envelope.fill(1.)
        envelope[step == 0] = 0.
        blocks = np.asarray([(i, min(end, i + args.subwindow), k)
                             for start, end, k in blocks
                             for i in range(start, end, args.subwindow)])
    elif args.short_release:
        for start, end, _ in blocks:
            if end - start > 1:
                if end - start > 3:
                    enabled[start + 1:end] = enabled[start + 1].copy()
                envelope[start + 2:end] = 1.
    if not args.compact:
        for start, end, _ in blocks:
            envelope[start] = 0.
            if tail:
                envelope[max(start, end - tail):end] = 0.
    train, val, test = split == 0, split == 1, split == 2
    state_mean = state[train].mean(0)
    state_scale = np.maximum(state[train].std(0), .01)
    state = (state - state_mean) / state_scale
    coord = np.maximum(np.sqrt(np.square(y[train]).mean(0)), .0005)
    group = np.ones(y.shape[1], np.float32)
    for sl in GROUPS.values():
        group[sl] = (sl.stop - sl.start)**-.5
        if args.group_scale:
            coord[sl] = max(float(np.sqrt(np.square(y[train, sl]).mean())), .0005)
    cuda = lambda v: torch.as_tensor(v, device="cuda")
    pose, delta, mask, scale = map(cuda, (state, displacement, enabled, envelope))
    motion = cuda(y / coord)
    weights = cuda(group)
    index = torch.arange(len(state), device="cuda")
    before = (index - 1).clamp_min(0)
    source = torch.cat((pose[before], motion.clamp(-8., 8.), (delta.flatten(1) * 40.).clamp(-4., 4.),
                        mask.to(pose.dtype)), 1)
    all_slots = torch.ones_like(mask)
    if args.search:
        source = source[:, :-8]
    torch.manual_seed(args.seed)
    hidden = args.hidden
    generator = Refiner(source.shape[1], args.angle, args.max_force, hidden).cuda()
    reader = nn.Sequential(nn.Linear(pose.shape[1] + 24, hidden), nn.SiLU(),
                           nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, y.shape[1])).cuda()

    def decode(rows, slots=None):
        slots = mask if slots is None else slots
        return generator(source[rows], delta[rows], slots[rows], scale[rows])

    def read(rows, force):
        return reader(torch.cat((pose[rows], force.flatten(1) * 2.), 1))

    optimizer = torch.optim.AdamW([*generator.parameters(), *reader.parameters()], lr=.001)
    tr, va = cuda(np.flatnonzero(train)), cuda(np.flatnonzero(val))
    best, weights_by_net, age = float("inf"), None, 0
    selected_blocks = cuda(blocks[split[blocks[:, 0]] != 2])
    reset = cuda(step == 0)
    for epoch in range(args.epochs):
        if args.search and epoch >= 20 and epoch % 12 == 8:
            with torch.no_grad():
                mask = select_windows(selected_blocks, mask, decode(index, all_slots),
                                      pose, motion, weights, reader, tail, args.compact, reset,
                                      args.subwindow, retain_common=args.retain_common)
        for rows in tr[torch.randperm(len(tr), device="cuda")].split(1024):
            force = decode(rows)
            err = ((read(rows, force) - motion[rows]) * weights).square().mean()
            change = force - decode(before[rows])
            loss = err + args.smooth * change.square().mean() + .001 * force.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = read(va, decode(va)).cpu().numpy() * coord
        score = np.mean([metrics(y[val, sl], pred[:, sl])["nrmse"] for sl in GROUPS.values()])
        if score < best - 1e-5:
            best, age = score, 0
            weights_by_net = [{k: v.detach().clone() for k, v in net.state_dict().items()}
                              for net in (generator, reader)]
        else:
            age += 1
            if age >= 16 and not args.search:
                break
    for net, w in zip((generator, reader), weights_by_net):
        net.load_state_dict(w)
    with torch.no_grad():
        if args.search:
            final_blocks = cuda(blocks)
            partitions = split[blocks[:, 0]] if args.marginal_counts else None
            mask = select_windows(final_blocks, mask, decode(index, all_slots),
                                  pose, motion, weights, reader, tail, args.compact, reset,
                                  args.subwindow, args.probe_steps, args.max_force,
                                  args.retain_common, partitions)
            blocks = final_blocks.cpu().numpy()
        force = decode(index)
    if args.inner_steps:
        with torch.no_grad():
            initial = generator.net(source)
        for round_id in range(args.refit_rounds + 1):
            reader.requires_grad_(False)
            force, initial = refine_parameters(generator, reader, initial, delta, mask, scale,
                                               pose, motion, weights, force, args.inner_steps)
            if round_id == args.refit_rounds:
                break
            reader.requires_grad_(True)
            opt = torch.optim.AdamW(reader.parameters(), lr=.0004)
            for _ in range(12):
                for rows in tr[torch.randperm(len(tr), device="cuda")].split(1024):
                    loss = ((read(rows, force[rows]) - motion[rows]) * weights).square().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
    with torch.no_grad():
        joint = read(index, force).cpu().numpy() * coord
        force = force.cpu().numpy()
    if args.max_force > 1.5:
        with np.load(base / "masked_spring" / "geometry.npz") as data:
            rotations = np.concatenate([data[f"r{c}"][step[clip == c]] for c in np.unique(clip)])
        world = force @ rotations.transpose(0, 2, 1)
        force = aligned_slew(world, clip, 3.) @ rotations
    norm = np.linalg.norm(force, axis=-1)
    assert (norm > 1e-5).sum(-1).max() <= 3 and norm.max() <= args.max_force + 1e-4
    assert ((force * displacement).sum(-1) >= -1e-7).all()
    channels = force.reshape(len(state), -1)
    force_mean = channels[train].mean(0)
    force_scale = np.maximum(channels[train].std(0), .10)
    x = np.c_[state, np.pad((channels - force_mean) / force_scale,
                           ((0, 0), (0, width - channels.shape[1])))].astype(np.float32)
    pred_test, predict, _ = fit(x, y, split, args.seed, path / "independent.pt")
    subset = test if args.test else val
    pred = pred_test if args.test else predict(x[val])
    result = dict(
        joint_readout={k: metrics(y[subset, sl], joint[subset, sl]) for k, sl in GROUPS.items()},
        independent={k: metrics(y[subset, sl], pred[:, sl]) for k, sl in GROUPS.items()},
        force=dict(mean_N=float(norm[subset].sum(-1).mean()),
                   p95_N=float(np.quantile(norm[subset].sum(-1), .95)),
                   active=np.bincount((norm[subset] > .05).sum(-1), minlength=4).tolist()),
        count_windows=np.bincount(blocks[subset[blocks[:, 0]], 2], minlength=4).tolist(),
    )
    torch.save(generator.state_dict(), path / "generator.pt")
    if args.export_solver:
        torch.save(dict(generator={k: v.cpu() for k, v in generator.state_dict().items()},
                        reader={k: v.cpu() for k, v in reader.state_dict().items()},
                        state_mean=state_mean.tolist(), state_scale=state_scale.tolist(),
                        force_mean=force_mean.tolist(), force_scale=force_scale.tolist(),
                        target_scale=coord.tolist(), group_weights=group.tolist(),
                        activity_scale=scale_by_group.tolist(), thresholds=thresholds.tolist(),
                        horizon=args.horizon, block=args.subwindow, hidden=hidden, force_width=width,
                        retain_common=args.retain_common, probe_steps=args.probe_steps,
                        inner_steps=args.inner_steps), path / "solver.pt")
    if args.test:
        np.savez_compressed(path / "test_predictions.npz", prediction=pred, target=y[test],
                            force=force[test], clip=clip[test])
        if args.angle == 0:
            np.savez_compressed(path / "aligned_forces.npz", force=force, blocks=blocks)
    (path / ("test.json" if args.test else "validation.json")).write_text(json.dumps(result, indent=2))
    print(args.seed, args.angle, best, result, flush=True)


if __name__ == "__main__":
    main()
