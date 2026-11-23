"""Decode dense spring prompts as noisy Cartesian displacement observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from dense_next import (
    DT, ROOT, POINT_CAP, TOTAL_CAP, SLEW_CAP, geometry_jacobians,
    limit_force, measure, noise_like, paired_ci, slew_force, summary, surface_points,
)
from pilot import SurfaceFK, inverse, load_windows, multiply, quat_apply, rotation_vector, yaw_quat


GAINS = (20, 40, 60, 80, 120, 160)
NOISE = (0.0, 0.1, 0.5)


def flatten_motion(windows, steps, device):
    motion = torch.as_tensor(windows, device=device)
    p, r, q = motion[..., :3], normalize(motion[..., 3:7], dim=-1), motion[..., 7:]
    delta = torch.cat((p.diff(dim=1), rotation_vector(multiply(r[:, 1:], inverse(r[:, :-1]))), q.diff(dim=1)), -1)
    heading = inverse(yaw_quat(r[:, steps]))
    poses = torch.cat((p[:, steps], multiply(heading, r[:, steps]), q[:, steps]), -1)
    return delta[:, steps].flatten(0, 1), delta[:, np.asarray(steps) - 1].flatten(0, 1), poses.flatten(0, 1), heading.flatten(0, 1)


def forces_and_jacobians(windows, steps, names, geometry, points, device):
    raw_forces, jacobians = [], []
    for chunk in np.array_split(windows, (len(windows) + 31) // 32):
        n, frames = chunk.shape[:2]
        flat = chunk.reshape(-1, 36)
        xyz, jac = geometry_jacobians(flat[:, 7:], flat[:, :3], flat[:, 3:7], names, points, geometry)
        xyz = torch.as_tensor(xyz.reshape(n, frames, -1, 3), device=device, dtype=torch.float32)
        raw_forces.append(xyz.diff(dim=1))
        selected = jac.reshape(n, frames, -1, 35)[:, steps]
        jacobians.append(torch.as_tensor(selected, device=device, dtype=torch.float32).flatten(0, 1))
    return torch.cat(raw_forces), torch.cat(jacobians)


def applied_force(displacement, gain, steps):
    desired = gain * displacement
    limited = limit_force(desired.flatten(0, 1), POINT_CAP, TOTAL_CAP).reshape_as(desired)
    current = torch.zeros_like(limited[:, 0])
    applied = []
    for target in limited.unbind(1):
        current = slew_force(current, target, DT, SLEW_CAP)
        applied.append(current)
    return torch.stack(applied, 1)[:, steps].flatten(0, 1).flatten(1)


def heading_frame(delta, orientation):
    return torch.cat((quat_apply(orientation, delta[:, :3]),
                      quat_apply(orientation, delta[:, 3:6]), delta[:, 6:]), -1)


def decode(jac, prior, scale, force, gain, regularizer):
    # Near saturation, the absolute scale is ambiguous. Retain the velocity
    # prior there. The guard uses only current geometry and observed history.
    observation_model = gain * jac
    expected = (observation_model @ prior.unsqueeze(-1)).squeeze(-1)
    magnitudes = expected.reshape(len(expected), -1, 3).norm(dim=-1)
    in_linear_range = (magnitudes.sum(-1) <= .9 * TOTAL_CAP) & (magnitudes.amax(-1) <= .9 * POINT_CAP)
    a = observation_model * scale
    at = a.transpose(-1, -2)
    residual = at @ (force - expected).unsqueeze(-1)
    identity = torch.eye(len(scale), device=jac.device)
    fitted = torch.linalg.solve(at @ a + regularizer**2 * identity, residual).squeeze(-1)
    return prior + scale * fitted * in_linear_range[:, None], in_linear_range


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout-seed", default=20261122, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed = 20261122
    device = torch.device(args.device)
    windows, rows, names = load_windows(ROOT / "dataset/compliance_mix_29h_v1", seed, 1, 260, True)
    stats = json.loads((ROOT / "artifacts/ceer_segmented_jacobian/geometry.json").read_text())
    points = surface_points(args.layout_seed)
    fk = SurfaceFK(stats, names, device)
    steps = list(range(6, windows.shape[1] - 1, 4))
    frame_rows = [row for row in rows for _ in steps]
    masks = {
        split: torch.tensor([row[0] == split for row in frame_rows], device=device)
        for split in ("train", "validation", "test")
    }
    d, prior, p, h = flatten_motion(windows, steps, device)
    prior_scale = (d[masks["train"]] - prior[masks["train"]]).std(dim=0).clamp_min(.00005)
    d_local = heading_frame(d, h)
    displacement, jac = forces_and_jacobians(windows, steps, names, stats, points, device)
    payload = {"layout_seed": args.layout_seed, "counts": {k: int(v.sum()) for k,v in masks.items()},
               "selection": "validation FK at component noise sigma 0.1 N", "spring": {}}
    valid, test = (masks[tag] for tag in ("validation", "test"))
    valid_args = (jac[valid], prior[valid], prior_scale)
    test_args = (jac[test], prior[test], prior_scale)
    categories = np.asarray([row[1] for row in frame_rows if row[0] == "test"])
    recordings = np.asarray([row[2] for row in frame_rows if row[0] == "test"])
    constant = measure(fk, p[test], d_local[test], heading_frame(prior[test], h[test])).cpu().numpy()
    payload["constant_velocity"] = summary(constant, categories)
    noise_val = noise_like(jac[valid, :, 0], seed + 72)
    noise_test = noise_like(jac[test, :, 0], seed + 73)
    results = {}
    for gain in GAINS:
        force = applied_force(displacement, gain, steps)
        valid_force = force[valid] + .1 * noise_val
        scores = {}
        for regularizer in (0.03, 0.06, 0.1, 0.2, 0.4):
            decoded, guard = decode(*valid_args, valid_force, gain, regularizer)
            err = measure(fk, p[valid], d_local[valid], heading_frame(decoded, h[valid]))
            scores[regularizer] = err[:, 0].mean().item()
        best = min(scores, key=scores.get)
        payload["spring"][gain] = {"validation_fk_mm": scores[best], "regularizer_N": best,
                                   "guarded_fraction_validation": (~guard).float().mean().item()}
        for noise in NOISE:
            decoded, _ = decode(*test_args, force[test] + noise * noise_test, gain, best)
            result = measure(fk, p[test], d_local[test], heading_frame(decoded, h[test])).cpu().numpy()
            results[gain, noise] = result[:, 0]
            payload["spring"][gain][f"noise_{noise:g}"] = summary(result, categories)
        print(gain, scores[best], [payload["spring"][gain][f"noise_{s:g}"]["all"]["mean_fk_mm"] for s in NOISE], flush=True)
    winner = min(GAINS, key=lambda gain: payload["spring"][gain]["validation_fk_mm"])
    payload["selected_gain_N_m"] = winner
    payload["improvement_ci95_mm_vs_velocity"] = paired_ci(results[winner, .1], constant[:, 0], recordings, seed)
    payload["improvement_ci95_mm_vs_other_gains"] = {
        k: paired_ci(results[winner, .1], results[k, .1], recordings, seed) for k in GAINS if k != winner
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
