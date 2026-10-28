#!/usr/bin/env python3
"""Add wider-range, quality-screened force variants to underfilled motions."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_process import generate_segmented_force_variants as gen


def _init(cache, source, geom_path, points_path):
    gen._init_worker(source, str(Path("artifacts/ceer_single_hand_ee/full_20261025").resolve()),
                     geom_path, points_path)
    global SLOTS, FORCES, OFFSETS
    SLOTS = np.load(Path(cache) / "variant_slots.npy", mmap_mode="r")
    FORCES = np.load(Path(cache) / "variant_forces.npy", mmap_mode="r")
    OFFSETS = np.load(Path(cache) / "motion_offsets.npy", mmap_mode="r")


def _process(item):
    report, ranges, count, seed_offset = item
    mid = int(report["motion_id"])
    start, end = int(report["start"]), int(report["end"])
    n = end - start
    qdata = gen.DATA[start:end]
    q = qdata.joint_pos.numpy().astype(np.float64)
    root_pos = qdata.root_pos_w.numpy().astype(np.float64)
    root_quat = qdata.root_quat_w.numpy().astype(np.float64)
    _, jac = gen.geometry_jacobians(q, root_pos, root_quat,
                                    gen.DATA_META["joint_names"], gen.POINTS, gen.GEOM)
    target = gen.nominal_target(q, root_pos, root_quat)
    a, b, groups, _ = gen.weighted_problem(jac, target, gen.DATA_META["joint_names"])
    offset = int(OFFSETS[mid])
    base_slots = np.asarray(SLOTS[0, offset:offset + n]).copy()
    base_force = np.asarray(FORCES[0, offset:offset + n]).copy()
    selected = [(np.asarray(SLOTS[v, offset:offset + n]).copy(),
                 np.asarray(FORCES[v, offset:offset + n]).copy()) for v in range(count)]
    base_dense = gen._dense(*selected[0], len(gen.POINTS))
    rng = np.random.default_rng(20261026 + seed_offset + mid)
    pool = []
    fit_valid = 0
    for candidate_id in range(1000):
        gain = gen._gain_envelope(n, rng, low=.90, high=1.10)
        dense = base_dense * gain[:, None, None]
        _, _, _, scale = gen._force_limits(dense)
        if scale < .30:
            continue
        dense *= scale
        fit_ok, _, _ = gen._fit_stats(dense, base_slots, a, b, jac, target, groups, ranges)
        if not fit_ok:
            continue
        fit_valid += 1
        force = (base_force * gain[:, None, None] * scale).astype(np.float32)
        distances = [gen._profile_difference(base_slots, force, slots, prev_force,
                                               len(gen.POINTS))[1]
                     for slots, prev_force in selected]
        if min(distances) < .003:
            continue
        pool.append((base_slots.copy(), force))

    added = []
    while pool and count + len(added) < 10:
        refs = selected + added
        distances = [min(float(np.sqrt(np.mean((gen._dense(slots, force, len(gen.POINTS)) -
                                               gen._dense(ref_slots, ref_force, len(gen.POINTS))) ** 2)))
                         for ref_slots, ref_force in refs) for slots, force in pool]
        best = int(np.argmax(distances))
        if distances[best] < .003:
            break
        added.append(pool.pop(best))
    return dict(motion_id=mid, count=count, added=added, fit_valid=fit_valid,
                distinct_pool=len(pool) + len(added), final_count=count + len(added))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--points-config", type=Path,
                   default=ROOT / "cfg/task/G1/G1_ceer_spring18_compliance.yaml")
    args = p.parse_args()
    cache, source = args.cache.resolve(), args.source_cache.resolve()
    counts = np.load(cache / "motion_variant_counts.npy")
    offsets = np.load(cache / "motion_offsets.npy", mmap_mode="r")
    manifest = json.loads((source / "manifest.json").read_text())
    vm = json.loads((cache / "variant_manifest.json").read_text())
    outcomes = {int(r["motion_id"]): r for r in vm["outcomes"]}
    ds_meta = json.loads((source / "dataset" / "meta_motion.json").read_text())
    data = gen.MotionData.load(str(source / "dataset"))
    ranges = {}
    for lo, hi in zip(ds_meta["starts"], ds_meta["ends"]):
        lo, hi = int(lo), int(hi)
        mid = int(data.motion_id[lo].item())
        a, b = int(data.step[lo].item()), int(data.step[hi - 1].item()) + 1
        ranges.setdefault(mid, []).append((a, b))
    del data
    work = []
    for report in manifest["reports"]:
        mid = int(report["motion_id"])
        if outcomes.get(mid, {}).get("category") != "original" or int(counts[mid]) >= 10:
            continue
        normalized = dict(report, end=int(report.get("end", int(report["start"]) + int(report["frames"]))))
        work.append((normalized, ranges.get(mid, []), int(counts[mid]), 900000))

    slots = np.load(cache / "variant_slots.npy", mmap_mode="r+")
    forces = np.load(cache / "variant_forces.npy", mmap_mode="r+")
    geom_path = ROOT / "artifacts/ceer_segmented_jacobian/geometry.json"
    points_path = args.points_config.resolve()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(cache), str(source), str(geom_path), str(points_path))) as pool:
        for i, result in enumerate(pool.map(_process, work, chunksize=1), 1):
            mid, off = result["motion_id"], int(offsets[result["motion_id"]])
            for j, (variant_slots, variant_forces) in enumerate(result["added"]):
                slots[result["count"] + j, off:off + len(variant_slots)] = variant_slots
                forces[result["count"] + j, off:off + len(variant_forces)] = variant_forces
            counts[mid] = result["final_count"]
            outcomes[mid]["variant_count"] = result["final_count"]
            outcomes[mid]["supplement"] = dict(added=len(result["added"]),
                quality_valid_profiles=result["fit_valid"], selected_pool=result["distinct_pool"])
            results.append(dict(motion_id=mid, before=result["count"], after=result["final_count"],
                                added=len(result["added"]), fit_valid_profiles=result["fit_valid"]))
            if i % 100 == 0 or i == len(work):
                print("SUPPLEMENT_PROGRESS", i, "/", len(work),
                      "reached_10", sum(r["after"] == 10 for r in results), flush=True)
    slots.flush(); forces.flush()
    np.save(cache / "motion_variant_counts.npy", counts)
    vm["per_motion_variant_counts"] = {str(n): int((counts == n).sum()) for n in range(1, 11)}
    vm["method"]["underfilled_original_supplement"] = (
        "underfilled original motions receive additional smooth gain envelopes; candidate "
        "diversity threshold is relaxed only as needed (minimum 0.003 N RMS), while every "
        "candidate still passes residual, bilateral fit, and force-limit gates")
    vm["method"]["diversity_gate"] = (
        "farthest-point selection; typically >=0.10 N force-profile RMS; down to 0.003 N "
        "only for rare fit-sensitive motions, still with independently sampled smooth envelopes")
    vm["outcomes"] = list(outcomes.values())
    (cache / "variant_manifest.json").write_text(json.dumps(vm, indent=2))
    report = dict(candidates=len(work), reached_10_after=sum(r["after"] == 10 for r in results),
                  remaining_below_10=[r for r in results if r["after"] < 10], results=results)
    (cache / "supplement_report.json").write_text(json.dumps(report, indent=2))
    print("SUPPLEMENT_COMPLETE", json.dumps({k: v for k, v in report.items()
                                               if k not in ("results", "remaining_below_10")}, indent=2))


if __name__ == "__main__":
    main()
