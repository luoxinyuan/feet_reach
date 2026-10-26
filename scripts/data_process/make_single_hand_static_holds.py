#!/usr/bin/env python3
"""Curate zero-force static hold clips from accepted CEER single-hand rollouts.

Each clip freezes a sampled, stable simulated pose from an already accepted
single-hand rollout. Body poses are copied from that frame, joint/root
velocities are zeroed, and the fixed EE reference is set to the sampled hand
poses. The force cache is generated separately with --zero-force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Accepted single-hand trajectory directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=150,
                        help="Static clip length at the source 50 Hz")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.count < 1 or args.frames < 50:
        parser.error("count must be positive and frames must be at least 50")

    source_summary = json.loads((source / "summary.json").read_text())
    candidates = [r for r in source_summary["records"] if r["accepted"]]
    if not candidates:
        raise RuntimeError("No accepted source trajectories")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(candidates), size=args.count, replace=args.count > len(candidates))
    raw = output / "raw"
    raw.mkdir(parents=True)
    records = []
    for out_i, pick in enumerate(picks):
        rec = candidates[int(pick)]
        with np.load(source / rec["file"], allow_pickle=True) as z:
            original = {k: z[k] for k in z.files}
        n = len(original["qpos"])
        lo, hi = min(10, n - 1), max(min(n - 10, n - 1), min(10, n - 1) + 1)
        frame = int(rng.integers(lo, hi)) if hi > lo else n // 2
        body_names = original["body_names"].tolist()
        body_ids = [body_names.index("left_hand_mimic"), body_names.index("right_hand_mimic")]

        # Freeze the full sampled simulator state. A zero-velocity static state
        # avoids teaching residual motion during an unforced hold.
        qpos = np.repeat(original["qpos"][frame:frame + 1].astype(np.float32), args.frames, axis=0)
        qvel = np.zeros((args.frames, original["qvel"].shape[-1]), dtype=np.float32)
        xpos = np.repeat(original["xpos"][frame:frame + 1].astype(np.float32), args.frames, axis=0)
        xquat = np.repeat(original["xquat"][frame:frame + 1].astype(np.float32), args.frames, axis=0)
        root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
        root_rotation = Rotation.from_quat(root_rot_xyzw)
        local_pos = np.einsum("tij,tbj->tbi", root_rotation.inv().as_matrix(),
                              xpos - qpos[:, None, :3]).astype(np.float32)
        body_rot = Rotation.from_quat(xquat[..., [1, 2, 3, 0]].reshape(-1, 4))
        local_rot_m = np.einsum("tij,tbjk->tbik", root_rotation.inv().as_matrix(),
                                body_rot.as_matrix().reshape(args.frames, len(body_names), 3, 3))
        local_rot = Rotation.from_matrix(local_rot_m.reshape(-1, 3, 3)).as_quat().reshape(
            args.frames, len(body_names), 4).astype(np.float32)
        hands_pos = local_pos[0, body_ids]
        hands_rot = Rotation.from_quat(local_rot[0, body_ids])
        target = np.concatenate((hands_pos.reshape(-1), hands_rot.as_rotvec().astype(np.float32).reshape(-1)))
        target = np.repeat(target[None, :], args.frames, axis=0).astype(np.float32)
        side = str(original["active_side"])
        name = f"static_{out_i:06d}_{side}.npz"
        np.savez_compressed(raw / name,
            fps=np.int32(50), qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat,
            root_pos=qpos[:, :3].astype(np.float32), root_rot=root_rot_xyzw.astype(np.float32),
            dof_pos=qpos[:, 7:].astype(np.float32), local_body_pos=local_pos,
            local_body_rot=local_rot,
            joint_names=original["joint_names"], body_names=original["body_names"],
            target_root_wrist_6d=target, active_side=np.asarray(side),
            quality=np.asarray([0., 0., 0., 0., float(qpos[0, 2]), 0.], dtype=np.float32))
        records.append(dict(index=out_i, active_side=side, active_hand_mae_m=0.,
            fixed_hand_mae_m=0., fixed_hand_max_error_m=0., max_root_drift_m=0.,
            min_root_height_m=float(qpos[0, 2]), max_tilt_rad=0., stable=True,
            tracking_ok=True, accepted=True, source_file=rec["file"], source_frame=frame,
            file=f"raw/{name}"))
        if (out_i + 1) % 500 == 0:
            print(f"STATIC_HOLDS {out_i + 1}/{args.count}", flush=True)

    summary = dict(seed=args.seed, count=args.count, envs=0, fps=50,
        duration_s=args.frames / 50, waypoints=0,
        controller="sampled accepted CEER rollout state, frozen with zero velocity",
        reference_pool=str(source), acceptance=dict(source="accepted stable and tracked rollouts",
            hold="all joints and body poses frozen; both hand targets equal sampled pose"),
        force_profile="zero force on every frame", by_side={}, accepted=args.count,
        records=records)
    for side in ("left", "right"):
        subset = [r for r in records if r["active_side"] == side]
        summary["by_side"][side] = dict(count=len(subset), accepted=len(subset),
            stable_fraction=1., tracking_ok_fraction=1., active_hand_mae_m=0., fixed_hand_mae_m=0.,
            max_root_drift_m=0.)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
