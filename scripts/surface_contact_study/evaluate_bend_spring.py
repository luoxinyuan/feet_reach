"""Probe a fitted back-spring force decoder on Crocoddyl reference motions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import TARGET_SCALE
from scripts.surface_contact_study.bend_spring import geometry, result_dir, spring
from active_adaptation.utils.segmented_guidance import geometry_jacobians


CASES = [
    ("crouch", "crouch", 0),
    ("forward", "difficulty_sweep/bend_forward_35deg", 1),
    ("backward", "large_bend_sweep/bend_backward_35deg", 1),
    ("left", "large_bend_sweep/bend_left_35deg", 0),
    ("right", "large_bend_sweep/bend_right_35deg", 0),
]
EXTRA_CASES = [
    ("crouch10", "difficulty_sweep/crouch_10cm", 0),
    ("crouch23", "difficulty_sweep/crouch_23cm", 0),
    ("forward20", "difficulty_sweep/bend_forward_20deg", 1),
    *((f"{side}15", f"difficulty_sweep/bend_{side}_15deg", int(side == "backward"))
      for side in ("backward", "left", "right")),
    *((f"{side}25", side, int(side == "backward")) for side in ("backward", "left", "right")),
    *((f"{side}45", f"large_bend_sweep/bend_{side}_45deg", int(side == "backward"))
      for side in ("backward", "left", "right")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--gain", type=int, required=True)
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--crouch-guard", action="store_true")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = parser.parse_args()
    if args.relative and args.crouch_guard:
        parser.error("relative springs do not use the crouch guard")
    if args.planar and not args.relative:
        parser.error("--planar requires --relative")
    torch.set_num_threads(2)
    base = args.input
    fit_dir = result_dir(base, args.gain, args.gate, args.crouch_guard, args.relative, args.planar)
    m = json.loads((Path(json.loads((base / "manifest.json").read_text())["source"]) /
                    "manifest.json").read_text())
    kinematics = json.loads((Path(m["force_cache"]) / "manifest.json").read_text())["geometry"]
    names = m["joint_names"]
    points = json.loads((base / "sparse_lookahead_spring" / "points.json").read_text())
    with np.load(fit_dir / "normalization.npz") as data:
        sm, ss, fm, fs = (data[k] for k in ("state_mean", "state_scale", "force_mean", "force_scale"))
    models = []
    for seed in args.seeds:
        weights = torch.load(fit_dir / f"model_{seed}.pt", weights_only=False)["weights"]
        width = weights["0.weight"].shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.SiLU(),
                                  torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 41))
        net.load_state_dict(weights)
        models.append(net.eval())
    results = []
    cases = CASES + EXTRA_CASES if args.sweep else CASES
    for name, reference, axis in cases:
        meta, d = load_arrays(args.references / reference / "motion")
        q = np.array(d["joint_pos"])[:, [meta["joint_names"].index(n) for n in names]]
        pos, quat = np.array(d["root_pos_w"]), np.array(d["root_quat_w"])
        g = geometry(q, pos, quat, names, points, kinematics, relative=args.relative)
        rotation = g[1]
        padded = [np.concatenate((v, np.repeat(v[-1:], 5, axis=0))) if v is not None else None
                  for v in g]
        world = spring(*padded, horizon=5, gain=args.gain, gate=args.gate,
                       crouch_guard=args.crouch_guard, relative=args.relative, planar=args.planar)
        root_force = world @ rotation
        state = np.c_[q, -rotation[:, 2, :], pos[:, 2]]

        def predict(force):
            adjusted = (force.reshape(len(q), -1) - fm) / fs
            force_channels = np.pad(adjusted, ((0, 0), (0, width - len(sm) - len(fm))))
            x = torch.from_numpy(np.c_[(state - sm) / ss, force_channels].astype(np.float32))
            with torch.inference_mode():
                return np.array([model(x).numpy() * TARGET_SCALE for model in models])

        preds = predict(root_force)
        zero = predict(np.zeros_like(root_force))
        a, b = 50, 200
        window = slice(a, b)
        norms = np.linalg.norm(world[window], axis=-1)
        if name.startswith("crouch"):
            gt = (pos[b, 2] - pos[a, 2]) * 100.
            values = [np.einsum("tij,stj->sti", rotation[window],
                                pred[:, window, 29:32])[:, :, 2].sum(-1) * 100.
                      for pred in (preds, zero)]
            torque_mean, consistent = None, None
        else:
            joint = names.index(("waist_roll_joint", "waist_pitch_joint")[axis])
            gt = np.degrees(q[b, joint] - q[a, joint])
            values = [np.degrees(pred[:, window, joint].sum(-1)) for pred in (preds, zero)]
            _, jac = geometry_jacobians(q, pos, quat, names, points, kinematics)
            torque = (jac[window, :, :, 6 + joint] * world[window]).sum((1, 2))
            delta = np.diff(q[a:b + 1, joint])
            meaningful = (np.abs(torque) > 1e-4) & (np.abs(delta) > 1e-5)
            torque_mean = float(torque.mean())
            consistent = (float((torque[meaningful] * delta[meaningful] > 0.).mean())
                          if meaningful.any() else None)
        result = dict(case=name, gt=float(gt), predicted=values[0].tolist(), ablated=values[1].tolist(),
                      active_fraction=(norms > .05).mean(0).tolist(),
                      mean_total_N=float(norms.sum(-1).mean()), peak_total_N=float(norms.sum(-1).max()),
                      mean_waist_torque_Nm=torque_mean, torque_direction_match=consistent)
        results.append(result)
        np.savez_compressed(fit_dir / f"{name}_trace.npz", force_world=world, force_root=root_force,
                            prediction=preds, zero_force_prediction=zero)
    filename = "crocoddyl_sweep.json" if args.sweep else "crocoddyl.json"
    (fit_dir / filename).write_text(json.dumps(results, indent=2))
    for r in results:
        print(r["case"], round(r["gt"], 2), np.round(r["predicted"], 2),
              np.round(r["ablated"], 2), r["mean_total_N"], r["mean_waist_torque_Nm"], flush=True)


if __name__ == "__main__":
    main()
