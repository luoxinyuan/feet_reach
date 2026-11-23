"""Audit forces and decoded motion for bend and yaw reference trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from active_adaptation.utils.segmented_guidance import geometry_jacobians, rot
from scripts.compliance_eval.protocol import load_arrays
from scripts.force_mapping_eval.evaluate_v2 import TARGET_SCALE
from scripts.force_mapping_eval.prepare_v2 import root_targets
from scripts.surface_contact_study.evaluate_bend_spring import CASES
from scripts.surface_contact_study.fixed8_learned_force import Refiner, refine_parameters, select_windows
from scripts.surface_contact_study.masked_spring import load_study


def readout(width, hidden, weights, device):
    net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                        nn.SiLU(), nn.Linear(hidden, 41)).to(device)
    net.load_state_dict(weights)
    return net.eval().requires_grad_(False)


def solve(path, state, target, displacement, device):
    saved = torch.load(path / "solver.pt", map_location="cpu", weights_only=True)
    to_tensor = lambda values: torch.as_tensor(values, dtype=torch.float32, device=device)
    pose = to_tensor((state - np.array(saved["state_mean"])) / saved["state_scale"])
    motion = to_tensor(target / saved["target_scale"])
    delta = to_tensor(displacement)
    weights = to_tensor(saved["group_weights"])
    previous = torch.arange(len(state), device=device).sub(1).clamp_min(0)
    source = torch.cat((pose[previous], motion.clamp(-8., 8.),
                        (delta.flatten(1) * 40.).clamp(-4., 4.)), 1)
    generator = Refiner(source.shape[1], 0, 1.5, saved["hidden"]).to(device)
    generator.load_state_dict(saved["generator"])
    generator.eval().requires_grad_(False)
    reader = readout(pose.shape[1] + 24, saved["hidden"], saved["reader"], device)
    size = saved["block"]
    target_energy = np.square(target / saved["activity_scale"]).sum(-1)
    blocks = []
    for start in range(0, len(state), size):
        end = min(start + size, len(state))
        count = 1 + np.searchsorted(saved["thresholds"], target_energy[start:end].mean())
        blocks.append((start, end, int(count)))
    envelope = torch.ones(len(state), device=device)
    envelope[0] = 0.
    mask = torch.ones(delta.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        initial = generator.net(source)
        dense = generator.force_from_parameters(initial, delta, mask, envelope)
        mask = select_windows(torch.as_tensor(blocks, device=device), mask, dense, pose,
                              motion, weights, reader, 0, compact=True, reset=0,
                              length=size, probe_steps=saved["probe_steps"],
                              retain_common=saved["retain_common"])
        prior = generator.force_from_parameters(initial, delta, mask, envelope)
    f, _ = refine_parameters(generator, reader, initial, delta, mask, envelope, pose, motion,
                             weights, prior, saved["inner_steps"])
    force = f.cpu().numpy()
    checkpoint = torch.load(path / "independent.pt", map_location="cpu", weights_only=False)
    probe = readout(pose.shape[1] + saved["force_width"], 256, checkpoint["weights"], device)

    def predict(input_force):
        x = (input_force.reshape(len(state), 24) - saved["force_mean"]) / saved["force_scale"]
        tokens = nn.functional.pad(to_tensor(x), (0, saved["force_width"] - 24))
        with torch.no_grad():
            return probe(torch.cat((pose, tokens), 1)).cpu().numpy() * TARGET_SCALE

    return force, predict(force), predict(np.zeros_like(force)), blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, geometry, points, _ = load_study(args.input)
    names = manifest["joint_names"]
    scenarios = [case for case in CASES if case[0] != "crouch"]
    scenarios.extend((f"yaw_{side}_{angle}", f"difficulty_sweep/yaw_{side}_{angle}deg", 2)
                     for angle in (15, 30) for side in ("left", "right"))
    output = args.model_dir / "rotation_audit" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for label, relative, axis in scenarios:
        metadata, d = load_arrays(args.references / relative / "motion")
        q = np.asarray(d["joint_pos"], dtype=np.float64)[:, [metadata["joint_names"].index(j) for j in names]]
        root, quat = (np.asarray(d[k], dtype=np.float64) for k in ("root_pos_w", "root_quat_w"))
        positions, jac = geometry_jacobians(q, root, quat, names, points, geometry)
        rotation = rot(quat)
        state = np.c_[q, -rotation[:, 2, :], root[:, 2]].astype(np.float32)
        padded_q, padded_root, padded_quat = (np.concatenate((v, np.repeat(v[-1:], 5, axis=0)))
                                               for v in (q, root, quat))
        target = np.c_[np.diff(padded_q, axis=0)[:len(q)],
                       root_targets(padded_root, padded_quat, 1)[:len(q)],
                       root_targets(padded_root, padded_quat, 5)].astype(np.float32)
        frames = np.arange(len(q))
        previous = np.maximum(0, frames - 1)
        nxt = np.minimum(previous + 2, len(q) - 1)
        disp = ((positions[nxt] - positions[previous]) @ rotation).astype(np.float32)
        f, prediction, ablated, blocks = solve(args.model_dir / f"seed{args.seed}", state,
                                               target, disp, "cuda")
        world = f @ rotation.transpose(0, 2, 1)
        norms = np.linalg.norm(world, axis=-1)
        active = norms > .05
        for lo, hi, k in blocks:
            assert (active[lo:hi].sum(-1) <= k).all()
        assert norms.max() <= 1.5001
        assert (f * disp).sum(-1).min() >= -1e-7
        assert np.linalg.norm(np.diff(world, axis=0), axis=-1).max() <= 3.0001
        a, b = 50, 200
        window = slice(a, b)
        if axis == 2:
            yaw = Rotation.from_matrix(rotation).as_euler("xyz")[:, 2]
            gt = float(np.rad2deg(np.unwrap(yaw)[b] - np.unwrap(yaw)[a]))
            increments = np.einsum("tij,tj->ti", rotation, target[:, 32:35])[:, 2]
            scores = [float(np.degrees(np.einsum("tij,tj->ti", rotation[window],
                                                 pred[window, 32:35])[:, 2].sum()))
                      for pred in (prediction, ablated)]
            per_point = np.cross(positions[previous] - root[previous, None], world)[:, :, 2]
        else:
            idx = names.index(("waist_roll_joint", "waist_pitch_joint")[axis])
            gt = float(np.degrees(q[b, idx] - q[a, idx]))
            increments = target[:, idx]
            scores = [float(np.degrees(pred[window, idx].sum())) for pred in (prediction, ablated)]
            per_point = (jac[previous, :, :, 6 + idx] * world).sum(-1)
        torque = per_point.sum(-1)
        significant = (np.abs(torque[window]) > 1e-4) & (np.abs(increments[window]) > 1e-5)
        correct = (torque[window] * increments[window])[significant] > 0.
        strongest = a + int(np.abs(torque[window]).argmax())
        result = dict(case=label, target_deg=gt, predicted_deg=scores[0], zero_force_deg=scores[1],
                      torque_sign_agreement=float(correct.mean()) if len(correct) else None,
                      torque_evaluated_frames=int(len(correct)),
                      mean_torque_Nm=float(torque[window].mean()), peak_torque_Nm=float(torque[strongest]),
                      peak_torque_force_root=f[strongest].tolist(), strongest_frame=strongest,
                      mean_torque_by_point_Nm=per_point[window].mean(0).tolist(),
                      active_fraction=active[window].mean(0).tolist(),
                      mean_total_N=float(norms[window].sum(-1).mean()),
                      peak_total_N=float(norms[window].sum(-1).max()),
                      counts=[k for lo, hi, k in blocks if lo < b and hi > a])
        results.append(result)
        np.savez_compressed(output / f"{label}.npz", force_world=world, force_root=f,
                            torque=torque, torque_per_point=per_point, position=positions,
                            prediction=prediction, zero_force_prediction=ablated, blocks=blocks)
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    for row in results:
        print(row["case"], row["counts"],
              (row["target_deg"], row["predicted_deg"], row["zero_force_deg"]),
              row["mean_torque_Nm"], row["torque_sign_agreement"], flush=True)


if __name__ == "__main__":
    main()
