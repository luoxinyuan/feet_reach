#!/usr/bin/env python3
"""Fit one-hand-only force prompts to CEER-generated single-hand trajectories.

With --report-only, this measures one-point fit quality on the accepted clips
without creating a cache. Otherwise it writes a MotionDataset and a segmented
guidance-cache-compatible slots/forces/motion_offsets bundle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.segmented_guidance import (
    geometry_jacobians, nominal_target, weighted_problem, optimize_forces,
)


def init_worker(geometry_path: str, points_path: str):
    global GEOM, POINTS
    torch.set_num_threads(1)
    GEOM = json.loads(Path(geometry_path).read_text())
    POINTS = yaml.safe_load(Path(points_path).read_text())["command"]["force_points"]


def fit_one(item):
    global GEOM, POINTS
    record, input_root = item
    path = Path(input_root) / record["file"]
    with np.load(path, allow_pickle=True) as z:
        qpos = z["qpos"].astype(np.float64)
        joint_names = z["joint_names"].tolist()
        side = str(z["active_side"])
    point_name = f"{side}_hand"
    point_id = [p["name"] for p in POINTS].index(point_name)
    q = qpos[:, 7:]
    root_pos = qpos[:, :3]
    root_quat = qpos[:, 3:7]
    points, jac = geometry_jacobians(q, root_pos, root_quat, joint_names,
                                     [POINTS[point_id]], GEOM)
    target = nominal_target(q, root_pos, root_quat)
    a, b, groups, _ = weighted_problem(jac, target, joint_names)
    slots = np.full((len(q), 3), -1, dtype=np.int16)
    slots[:, 0] = 0  # local point zero maps to the active hand only
    force, residual, qp = optimize_forces(a, b, slots)
    local_force = force[:, 0]
    predicted = np.einsum("tpjd,tpj->td", jac, local_force[:, None])
    arm_errors = []
    for ids in groups[3:5]:
        err = np.linalg.norm(predicted[:, ids] - target[:, ids], axis=1)
        den = np.maximum(np.linalg.norm(target[:, ids], axis=1), .3)
        arm_errors.append(err / den)
    nonzero = np.linalg.norm(local_force, axis=-1) > 1e-4
    out = dict(index=int(record["index"]), side=side, input_file=record["file"],
        point=point_name, point_id=point_id, frames=int(len(q)),
        fit_mean=float(residual.mean()), fit_p95=float(np.quantile(residual, .95)),
        arm_fit_mean=[float(x.mean()) for x in arm_errors],
        force_peak_N=float(np.linalg.norm(local_force, axis=-1).max()),
        force_mean_N=float(np.linalg.norm(local_force, axis=-1).mean()),
        active_force_fraction=float(nonzero.mean()),
        qp_iterations=int(qp["qp_iterations"]), numerical_scale=float(qp["numerical_scale"]))
    out["force_fit_ok"] = (out["fit_mean"] <= .85 and max(out["arm_fit_mean"]) <= .85
                            and out["active_force_fraction"] >= .5)
    return out, slots, force


def direct_profile(record, input_root, points, seed):
    """Create a smooth one-hand force input in the live root frame."""
    path = Path(input_root) / record["file"]
    with np.load(path, allow_pickle=True) as z:
        qpos = z["qpos"].astype(np.float64)
        side = str(z["active_side"])
    n = len(qpos)
    names = [p["name"] for p in points]
    point_name = f"{side}_ee" if f"{side}_ee" in names else f"{side}_hand"
    point_id = names.index(point_name)
    rng = np.random.default_rng(seed + int(record["index"]))
    direction_b = np.eye(3)[rng.integers(0, 3)] * rng.choice([-1., 1.])
    direction_b += rng.normal(0., .22, 3)
    direction_b /= np.linalg.norm(direction_b)
    magnitude = float(rng.uniform(5., 20.))
    ramp = int(rng.integers(18, 27))
    start = int(rng.integers(15, 36))
    hold = int(rng.integers(30, 66))
    stop = min(n - 1, start + ramp + hold)
    scalar = np.zeros(n, dtype=np.float64)
    rise_end = min(n, start + ramp)
    u = np.arange(max(0, rise_end - start), dtype=np.float64) / max(ramp, 1)
    scalar[start:rise_end] = magnitude * (10*u**3 - 15*u**4 + 6*u**5)
    scalar[rise_end:stop] = magnitude
    fall_end = min(n, stop + ramp)
    m = max(0, fall_end - stop)
    if m:
        u = 1. - np.arange(m, dtype=np.float64) / max(ramp, 1)
        scalar[stop:fall_end] = magnitude * (10*u**3 - 15*u**4 + 6*u**5)
    root_rot = Rotation.from_quat(qpos[:, 3:7][:, [1, 2, 3, 0]])
    world_dir = root_rot.apply(np.broadcast_to(direction_b, (n, 3)))
    slots = np.full((n, 3), -1, dtype=np.int16)
    slots[:, 0] = point_id
    forces = np.zeros((n, 3, 3), dtype=np.float32)
    forces[:, 0] = (world_dir * scalar[:, None]).astype(np.float32)
    norms = np.linalg.norm(forces[:, 0], axis=-1)
    slew = np.linalg.norm(np.diff(forces[:, 0], axis=0), axis=-1) / .02
    report = dict(index=int(record["index"]), side=side, input_file=record["file"],
        point=point_name, point_id=point_id, frames=n,
        profile="single smooth pulse, fixed root-frame direction", force_peak_N=float(norms.max()),
        force_mean_N=float(norms.mean()), active_force_fraction=float((norms > 1e-4).mean()),
        max_slew_N_s=float(slew.max(initial=0.)), force_fit_ok=True)
    if report["force_peak_N"] > 30.0001 or report["max_slew_N_s"] > 150.0001:
        raise RuntimeError(f"Generated force violates constraints: {report}")
    return report, slots, forces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--direct-profile", action="store_true",
                   help="Use smooth commanded force profiles rather than a one-point inverse-Jacobian fit")
    p.add_argument("--zero-force", action="store_true",
                   help="Create an all-zero-force cache for curated static hold clips")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.zero_force and not args.direct_profile:
        p.error("--zero-force requires --direct-profile")
    source = args.input.resolve()
    summary = json.loads((source / "summary.json").read_text())
    records = [r for r in summary["records"] if r["accepted"]]
    if args.limit:
        records = records[:args.limit]
    out = args.output.resolve()
    if not args.report_only:
        out.mkdir(parents=True, exist_ok=False)
    point_cfg = ROOT / "cfg/task/G1/G1_ceer_spring18_compliance.yaml"
    geometry_path = ROOT / "artifacts/ceer_segmented_jacobian/geometry.json"
    init_worker(str(geometry_path), str(point_cfg))
    POINTS = yaml.safe_load(point_cfg.read_text())["command"]["force_points"]
    results = []
    cached = []
    if args.direct_profile:
        for i, record in enumerate(records, 1):
            if args.zero_force:
                with np.load(source / record["file"], allow_pickle=True) as z:
                    side = str(z["active_side"])
                    n = len(z["qpos"])
                slots = np.full((n, 3), -1, dtype=np.int16)
                force = np.zeros((n, 3, 3), dtype=np.float32)
                report = dict(index=int(record["index"]), side=side,
                    input_file=record["file"], point="none", point_id=-1, frames=n,
                    profile="static hold, zero force", force_peak_N=0., force_mean_N=0.,
                    active_force_fraction=0., max_slew_N_s=0., force_fit_ok=True)
            else:
                report, slots, force = direct_profile(record, str(source), POINTS, args.seed)
            results.append(report)
            cached.append((report, slots, force))
            if i % 250 == 0 or i == len(records):
                print("DIRECT_FORCE", i, "/", len(records), flush=True)
    else:
        items = [(r, str(source)) for r in records]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(str(geometry_path), str(point_cfg))) as pool:
            for i, pair in enumerate(pool.map(fit_one, items, chunksize=1), 1):
                report, slots, force = pair
                results.append(report)
                if report["force_fit_ok"]:
                    cached.append((report, slots, force))
                if i % 25 == 0 or i == len(items):
                    print("FORCE_FIT", i, "/", len(items), "fit_ok",
                          sum(r["force_fit_ok"] for r in results), flush=True)
    by_side = {}
    for side in ("left", "right"):
        subset = [r for r in results if r["side"] == side]
        by_side[side] = dict(count=len(subset), fit_ok=sum(r["force_fit_ok"] for r in subset),
            fit_mean=float(np.mean([r["fit_mean"] for r in subset])) if subset and "fit_mean" in subset[0] else None,
            arm_fit_mean=np.mean([r["arm_fit_mean"] for r in subset], axis=0).tolist() if subset and "arm_fit_mean" in subset[0] else None,
            force_peak_p95_N=float(np.quantile([r["force_peak_N"] for r in subset], .95)) if subset else None,
            active_force_fraction=float(np.mean([r["active_force_fraction"] for r in subset])) if subset else None)
    report = dict(input=str(source), candidates=len(records), fit_ok=sum(r["force_fit_ok"] for r in results),
        frame_rate_hz=50, method=("curated static hold clips; all force slots disabled"
            if args.zero_force else "procedural smooth single-hand pulse; 5–20 N; root-frame direction; 30 N point/60 N total/150 N/s constraints"
            if args.direct_profile else "reference-only nominal target: 80*joint_delta, root reference feedforward; fixed active-hand force point; 30N point/60N total/150N/s constraints"),
        by_side=by_side, records=results)
    report_path = (out / "force_fit_report.json") if not args.report_only else (source / "force_fit_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print("FIT_SUMMARY", json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
    if args.report_only:
        return
    if not cached:
        raise RuntimeError("No clips passed the selected force generation quality gate")

    accepted_raw = out / "accepted_raw"
    accepted_raw.mkdir()
    # Keep force-fit-approved files only, in precisely the same order used below.
    files = []
    for report, _, _ in cached:
        src = source / report["input_file"]
        dst = accepted_raw / f"clip_{len(files):06d}.npz"
        with np.load(src, allow_pickle=True) as z:
            arrays = {k: z[k] for k in z.files}
        qpos = arrays["qpos"].astype(np.float32)
        root_rot = qpos[:, 3:7][:, [1, 2, 3, 0]]
        rr = Rotation.from_quat(root_rot)
        xpos = arrays["xpos"].astype(np.float32)
        xquat = arrays["xquat"].astype(np.float32)
        body_quat_xyzw = xquat[..., [1, 2, 3, 0]]
        body_rot_m = Rotation.from_quat(body_quat_xyzw.reshape(-1, 4)).as_matrix().reshape(
            len(qpos), len(arrays["body_names"]), 3, 3)
        local_rot_m = np.einsum("tij,tbjk->tbik", rr.inv().as_matrix(), body_rot_m)
        local_pos = np.einsum("tij,tbj->tbi", rr.inv().as_matrix(), xpos - qpos[:, None, :3])
        arrays.update(root_pos=qpos[:, :3], root_rot=root_rot, dof_pos=qpos[:, 7:],
            local_body_pos=local_pos.astype(np.float32),
            local_body_rot=Rotation.from_matrix(local_rot_m.reshape(-1, 3, 3)).as_quat().reshape(
                len(qpos), len(arrays["body_names"]), 4).astype(np.float32))
        np.savez_compressed(dst, **arrays)
        files.append(dst)
    dataset_path = out / "dataset"
    data, meta = MotionDataset.create_from_path(
        str(accepted_raw), target_fps=50, mem_path=str(dataset_path), segment_len=1000,
        storage_float_dtype=torch.float16)
    dataset_meta = json.loads((dataset_path / "meta_motion.json").read_text())
    total = int(sum(len(x[1]) for x in cached))
    slots_mm = np.lib.format.open_memmap(out / "slots.npy", mode="w+", dtype=np.int16, shape=(total, 3))
    forces_mm = np.lib.format.open_memmap(out / "forces.npy", mode="w+", dtype=np.float32, shape=(total, 3, 3))
    motion_offsets = np.full(len(cached), -1, dtype=np.int64)
    frame_cursor = 0
    cache_reports = []
    for motion_id, (r, slots, force) in enumerate(cached):
        n = len(slots)
        slots = slots.copy()
        slots[slots >= 0] = r["point_id"]
        slots_mm[frame_cursor:frame_cursor+n] = slots
        forces_mm[frame_cursor:frame_cursor+n] = force
        motion_offsets[motion_id] = frame_cursor
        frame_cursor += n
        cache_reports.append(dict(motion_id=motion_id, start=int(dataset_meta["starts"][motion_id]),
            end=int(dataset_meta["ends"][motion_id]), **r))
    slots_mm.flush(); forces_mm.flush()
    np.save(out / "motion_offsets.npy", motion_offsets)
    source_points = POINTS
    manifest = dict(split="train", clips=len(cached), frames=frame_cursor,
        accepted_frames=frame_cursor, accepted_spans=len(cached),
        point_frames=[int(sum(report["frames"] for report in results
                              if report["force_fit_ok"] and report["point_id"] == i))
                      for i in range(len(source_points))],
        points=source_points, geometry_sha256=GEOM["sha256"], reports=cache_reports,
        constraints=dict(per_point_N=30, total_N=60, slew_N_per_s=150,
                         max_points=1, force_point="active-side hand only"),
        target=("static two-hand EE hold; no force guidance" if args.zero_force else
                "80 joint delta; reference-only root translation/yaw feedforward; no actual-state feedback"),
        force_fit=(dict(method="all-zero-force static holds", force_ok_clips=len(cached)) if args.zero_force else
                   dict(method="fixed active-hand point", mean_error=float(np.mean([r["fit_mean"] for r,_,_ in cached])),
                        force_ok_clips=len(cached)) if "fit_mean" in cached[0][0]
                   else dict(method="procedural smooth force profile", seed=args.seed,
                             max_slew_N_s=float(max(r["max_slew_N_s"] for r,_,_ in cached)),
                             force_ok_clips=len(cached)))
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("CACHE_COMPLETE", json.dumps({k: v for k, v in manifest.items() if k != "reports"}), flush=True)


if __name__ == "__main__":
    main()
