#!/usr/bin/env python3
"""Generate several validated, deterministic force profiles per training motion.

Variant 0 preserves the current screened cache exactly. Original motion clips get
near-optimal alternative point layouts and, when needed, alternate smooth QP fits.
Accepted single-hand EE motions get alternate smooth procedural pulses. Static
holds remain zero-force. The result stores a small per-motion variant count so
the runtime can sample one complete profile at episode reset.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
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
from scipy.ndimage import gaussian_filter1d
from active_adaptation.utils.motion import MotionData
from active_adaptation.utils.segmented_guidance import (
    geometry_jacobians,
    nominal_target,
    optimize_forces,
    select_segments,
    weighted_problem,
)
from scripts.data_process.prepare_single_hand_ee_guidance import direct_profile


def _init_worker(cache: str, source: str, geom_path: str, points_path: str):
    global DATA, DATA_META, GEOM, POINTS, SOURCE
    torch.set_num_threads(1)
    DATA = MotionData.load(str(Path(cache) / "dataset"))
    DATA_META = json.loads((Path(cache) / "dataset" / "meta_motion.json").read_text())
    GEOM = json.loads(Path(geom_path).read_text())
    POINTS = yaml.safe_load(Path(points_path).read_text())["command"]["force_points"]
    SOURCE = Path(source)


def _dense(slots, force, npoints):
    out = np.zeros((len(slots), npoints, 3), dtype=np.float32)
    for k in range(slots.shape[1]):
        mask = slots[:, k] >= 0
        if mask.any():
            out[np.where(mask)[0], slots[mask, k]] = force[mask, k]
    return out


def _passes_training_spans(slots, residual, arm_errors, ranges):
    """Require every currently sampled subspan to retain the existing fit gate."""
    boundaries = np.r_[0, np.where(np.any(slots[1:] != slots[:-1], axis=1))[0] + 1, len(slots)]
    for lo, hi in ranges:
        if hi <= lo:
            continue
        cuts = boundaries[(boundaries > lo) & (boundaries < hi)]
        pieces = np.r_[lo, cuts, hi]
        for a, b in zip(pieces[:-1], pieces[1:]):
            if residual[a:b].mean() > .85:
                return False
            if max(err[a:b].mean() for err in arm_errors) > .85:
                return False
    return True


def _profile_difference(slots, force, previous_slots, previous_force, npoints):
    layout_fraction = float(np.mean(np.any(slots != previous_slots, axis=1)))
    d0 = _dense(slots, force, npoints)
    d1 = _dense(previous_slots, previous_force, npoints)
    force_rmse = float(np.sqrt(np.mean((d0 - d1) ** 2)))
    return layout_fraction, force_rmse


def _gain_envelope(n, rng, low=.62, high=1.32):
    knots=int(rng.integers(4,10))
    x=np.linspace(0,n-1,knots)
    y=rng.uniform(low,high,knots)
    gain=np.interp(np.arange(n),x,y)
    return gaussian_filter1d(gain,sigma=float(rng.uniform(5.,18.)),mode='nearest')


def _fit_stats(dense, slots, a, b, jac, target, groups, ranges):
    pred=np.einsum('tdi,ti->td',a,dense.reshape(len(dense),-1))
    residual=np.linalg.norm(pred-b,axis=1)/np.maximum(np.linalg.norm(b,axis=1),.2)
    actual=np.einsum('tpjd,tpj->td',jac,dense)
    arms=[np.linalg.norm(actual[:,ids]-target[:,ids],axis=1)
          /np.maximum(np.linalg.norm(target[:,ids],axis=1),.3) for ids in groups[3:5]]
    return _passes_training_spans(slots,residual,arms,ranges),residual,arms


def _force_limits(dense):
    peak=float(np.linalg.norm(dense,axis=-1).max(initial=0.))
    budget=float(np.linalg.norm(dense,axis=-1).sum(-1).max(initial=0.))
    slew=float(np.linalg.norm(np.diff(dense,axis=0),axis=-1).max(initial=0.)/.02)
    scale=min(1.,30/max(peak,1e-9),60/max(budget,1e-9),150/max(slew,1e-9))
    return peak,budget,slew,scale


def _work(item):
    report, ranges, baseline_slots, baseline_force = item
    mid = int(report["motion_id"])
    start, end = int(report["start"]), int(report["end"])
    n = end - start
    qdata = DATA[start:end]
    if not torch.all(qdata.motion_id == mid):
        raise RuntimeError(f"motion_id mismatch in cache span: expected {mid}")
    category = "static" if report.get("profile", "").startswith("static hold") else (
        "single_hand" if "side" in report and "index" in report else "original")
    variants = [(baseline_slots.copy(), baseline_force.copy(), "existing_cache")]
    metrics = []

    if category == "original":
        q = qdata.joint_pos.numpy().astype(np.float64)
        root_pos = qdata.root_pos_w.numpy().astype(np.float64)
        root_quat = qdata.root_quat_w.numpy().astype(np.float64)
        _, jac = geometry_jacobians(q, root_pos, root_quat, DATA_META["joint_names"], POINTS, GEOM)
        target = nominal_target(q, root_pos, root_quat)
        a, b, groups, _ = weighted_problem(jac, target, DATA_META["joint_names"])
        # First add distinct near-optimal point-layout alternatives when they pass
        # the same tracking-fit gate on every currently sampled span.
        for name, point_cost, switch_cost in [("compact_layout",.065,.65),
                                               ("flexible_layout",.015,.15)]:
            slots,_=select_segments(a,b,len(POINTS),point_cost=point_cost,switch_cost=switch_cost)
            force,residual,qp=optimize_forces(a,b,slots)
            dense=_dense(slots,force,len(POINTS))
            fit_ok,residual,arms=_fit_stats(dense,slots,a,b,jac,target,groups,ranges)
            peak,budget,slew,scale=_force_limits(dense)
            distinct=min(_profile_difference(slots,force,s0,f0,len(POINTS))
                         for s0,f0,_ in variants)
            if fit_ok and scale>=.999 and (distinct[0]>=.05 or distinct[1]>=.10):
                variants.append((slots.astype(np.int16),force.astype(np.float32),name))
                metrics.append(dict(name=name,accepted=True,fit_mean=float(residual.mean()),
                    arm_fit_mean=[float(x.mean()) for x in arms],force_rmse_N=distinct[1],
                    layout_change_fraction=distinct[0],max_point_N=peak,max_total_N=budget,
                    max_slew_N_s=slew,qp_iterations=int(qp['qp_iterations'])))
            else:
                metrics.append(dict(name=name,accepted=False,reason='fit_or_diversity_gate'))

        # Add smoothly varying force-strength envelopes to expose the student to
        # different plausible loads along the same motion. Keep only profiles that
        # pass the existing reconstruction gate and physical force limits.
        rng=np.random.default_rng(20261026+mid)
        base_dense=_dense(baseline_slots,baseline_force,len(POINTS))
        bank=list(variants)
        accepted_envelopes=0
        for envelope_id in range(120):
            gain=_gain_envelope(n,rng)
            dense=base_dense*gain[:,None,None]
            peak,budget,slew,scale=_force_limits(dense)
            if scale<.55:
                continue
            dense*=scale
            fit_ok,residual,arms=_fit_stats(dense,baseline_slots,a,b,jac,target,groups,ranges)
            if not fit_ok:
                continue
            force=baseline_force*gain[:,None,None]*scale
            candidate=(baseline_slots.astype(np.int16),force.astype(np.float32),f'gain_envelope_{envelope_id:03d}')
            if min(_profile_difference(candidate[0],candidate[1],s0,f0,len(POINTS))[1]
                   for s0,f0,_ in bank)<.05:
                continue
            bank.append(candidate);accepted_envelopes+=1

        # Greedy farthest-point selection yields up to nine genuinely different
        # force signals (RMS separation >= 0.10 N) plus the preserved baseline.
        selected=[bank[0]]
        remaining=bank[1:]
        while remaining and len(selected)<10:
            distances=[min(_profile_difference(s,f,ps,pf,len(POINTS))[1]
                           for ps,pf,_ in selected) for s,f,_ in remaining]
            best=int(np.argmax(distances))
            if distances[best]<.10:break
            selected.append(remaining.pop(best))
        variants=selected
        metrics.append(dict(name='smooth_gain_envelopes',attempted=120,
                            quality_and_force_valid=accepted_envelopes,
                            selected_gain_profiles=sum(name.startswith('gain_envelope_')
                                                       for _,_,name in variants),
                            selected_total=len(variants),
                            min_pairwise_force_rmse_N=min(
                                _profile_difference(s,f,ps,pf,len(POINTS))[1]
                                for i,(s,f,_) in enumerate(variants)
                                for ps,pf,_ in variants[:i]) if len(variants)>1 else 0.))
    elif category == "single_hand":
        for seed in range(20261031,20261040):
            name=f'pulse_seed_{seed}'
            item = dict(index=int(report["index"]), file=report["input_file"])
            profile, slots, force = direct_profile(item, str(SOURCE), POINTS, seed)
            dense = _dense(slots, force, len(POINTS))
            peak = float(np.linalg.norm(dense, axis=-1).max(initial=0.))
            budget = float(np.linalg.norm(dense, axis=-1).sum(-1).max(initial=0.))
            slew = float(np.linalg.norm(np.diff(dense, axis=0), axis=-1).max(initial=0.) / .02)
            distinct = min(_profile_difference(slots, force, s0, f0, len(POINTS))
                           for s0, f0, _ in variants)
            if peak > 30.001 or budget > 60.001 or slew > 150.01:
                metrics.append(dict(name=name, accepted=False, reason="force_constraint"))
                continue
            if distinct[0] < .05 and distinct[1] < .10:
                metrics.append(dict(name=name, accepted=False, reason="not_distinct",
                                    layout_change_fraction=distinct[0], force_rmse_N=distinct[1]))
                continue
            variants.append((slots.astype(np.int16), force.astype(np.float32), name))
            metrics.append(dict(name=name, accepted=True, peak_N=peak, total_N=budget,
                                max_slew_N_s=slew, active_fraction=profile["active_force_fraction"],
                                layout_change_fraction=distinct[0], force_rmse_N=distinct[1]))

    return dict(motion_id=mid, start=start, end=end, category=category,
                variants=variants, candidates=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--single-hand-source", type=Path,
                   default=Path("artifacts/ceer_single_hand_ee/full_20261025"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    source = args.source_cache.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((source / "manifest.json").read_text())
    base_slots = np.load(source / "slots.npy", mmap_mode="r")
    base_force = np.load(source / "forces.npy", mmap_mode="r")
    motion_offsets = np.load(source / "motion_offsets.npy", mmap_mode="r")
    reports = manifest["reports"]
    # Map currently active training spans to the corresponding original frame ranges.
    ds_meta = json.loads((source / "dataset" / "meta_motion.json").read_text())
    data = MotionData.load(str(source / "dataset"))
    active_ranges = {}
    for lo, hi in zip(ds_meta["starts"], ds_meta["ends"]):
        lo, hi = int(lo), int(hi)
        mid = int(data.motion_id[lo].item())
        step0 = int(data.step[lo].item())
        step1 = int(data.step[hi - 1].item()) + 1
        active_ranges.setdefault(mid, []).append((step0, step1))
    del data

    out.mkdir(parents=True)
    (out / "dataset").symlink_to((source / "dataset").resolve(), target_is_directory=True)
    (out / "motion_offsets.npy").symlink_to((source / "motion_offsets.npy").resolve())
    (out / "manifest.json").symlink_to((source / "manifest.json").resolve())
    nframes = len(base_slots)
    nvariants = 10
    slots_mm = np.lib.format.open_memmap(out / "variant_slots.npy", mode="w+", dtype=np.int16,
                                         shape=(nvariants, nframes, 3))
    force_mm = np.lib.format.open_memmap(out / "variant_forces.npy", mode="w+", dtype=np.float32,
                                         shape=(nvariants, nframes, 3, 3))
    for vi in range(nvariants):
        slots_mm[vi] = base_slots
        force_mm[vi] = base_force
    counts = (motion_offsets >= 0).astype(np.uint8)
    by_mid = {int(r["motion_id"]): r for r in reports}
    work = []
    for r in reports:
        mid = int(r["motion_id"])
        start = int(r["start"])
        end = int(r.get("end", start + int(r["frames"])))
        off = int(motion_offsets[mid])
        if off < 0 or end - start != int(r["frames"]):
            raise RuntimeError(f"Bad force/motion offset for motion {mid}")
        # `start`/`end` describe the raw concatenated dataset range and match force cache frames.
        frame_off = off
        length = end - start
        sl = np.asarray(base_slots[frame_off:frame_off + length]).copy()
        fo = np.asarray(base_force[frame_off:frame_off + length]).copy()
        category = "static" if r.get("profile", "").startswith("static hold") else (
            "single_hand" if "side" in r and "index" in r else "original")
        if category == "original":
            # Original reports carry the raw data start; their cache offset is at step zero.
            normalized = dict(r, end=end)
            work.append((normalized, active_ranges.get(mid, []), sl, fo))
        elif category == "single_hand":
            normalized = dict(r, end=end)
            work.append((normalized, [], sl, fo))
        else:
            counts[mid] = 1

    # Process all non-static clips. Every result always keeps baseline scheme zero.
    outcomes = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(str(source), str(args.single_hand_source.resolve()),
                                       str(ROOT / "artifacts/ceer_segmented_jacobian/geometry.json"),
                                       str(ROOT / "cfg/task/G1/G1_ceer_spring18_compliance.yaml"))) as pool:
        for i, result in enumerate(pool.map(_work, work, chunksize=1), 1):
            mid = result["motion_id"]
            off = int(motion_offsets[mid])
            for vi, (slots, forces, _) in enumerate(result["variants"]):
                length = len(slots)
                slots_mm[vi, off:off + length] = slots
                force_mm[vi, off:off + length] = forces
            counts[mid] = len(result["variants"])
            outcomes.append(dict(motion_id=mid, category=result["category"],
                                 variant_count=len(result["variants"]),
                                 accepted_candidates=result["candidates"]))
            if i % 250 == 0 or i == len(work):
                print("VARIANT_PROGRESS", i, "/", len(work), "with_multiple",
                      sum(x["variant_count"] > 1 for x in outcomes), flush=True)
    slots_mm.flush(); force_mm.flush()
    np.save(out / "motion_variant_counts.npy", counts)
    variant_counts = {str(n): int((counts == n).sum()) for n in range(1, nvariants + 1)}
    summary = dict(source_cache=str(source), clips=len(reports), frames=nframes,
                    variants_max=nvariants, per_motion_variant_counts=variant_counts,
                    accepted_spans=manifest["accepted_spans"], accepted_frames=manifest["accepted_frames"],
                    constraints=manifest["constraints"], method={
                    "original": "baseline plus quality-gated alternate layouts and 120 smooth gain envelopes; farthest-point selection up to 10",
                    "single_hand": "baseline plus deterministic smooth pulses with seeds 20261031 through 20261039",
                        "static": "single zero-force profile",
                        "selection": "random per environment reset, uniform over that motion's valid variants",
                        "quality_gate": "all active training spans retain per-segment residual and both-arm mean <= 0.85",
                        "diversity_gate": "pairwise force-profile RMS separation >=0.10 N",
                    }, outcomes=outcomes)
    (out / "variant_manifest.json").write_text(json.dumps(summary, indent=2))
    print("VARIANT_CACHE_COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    main()
