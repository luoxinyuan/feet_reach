#!/usr/bin/env python3
"""Append single-hand EE trajectories and force prompts to a CEER segmented cache.

The existing cache indexes forces by the raw motion_id stored in its MotionData.
This merger preserves those IDs, assigns the appended trajectories fresh IDs, and
keeps the original accepted motion spans while adding all single-hand clips.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tensordict import MemoryMappedTensor

from active_adaptation.utils.motion import MotionData


def read_json(path: Path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--augment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base, aug, out = args.base.resolve(), args.augment.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    ds_out = out / "dataset"
    ds_out.mkdir()

    base_manifest = read_json(base / "manifest.json")
    aug_manifest = read_json(aug / "manifest.json")
    if base_manifest["points"] != aug_manifest["points"]:
        raise ValueError("Force point layouts differ")
    if base_manifest["geometry_sha256"] != aug_manifest["geometry_sha256"]:
        raise ValueError("Robot geometry hashes differ")

    base_data = MotionData.load(str(base / "dataset"))
    aug_data = MotionData.load(str(aug / "dataset"))
    base_meta = read_json(base / "dataset" / "meta_motion.json")
    aug_meta = read_json(aug / "dataset" / "meta_motion.json")
    if base_meta["body_names"] != aug_meta["body_names"] or base_meta["joint_names"] != aug_meta["joint_names"]:
        raise ValueError("Motion body/joint layouts differ")
    if set(base_meta["info"]) != set(aug_meta["info"]):
        raise ValueError("Motion extra-info schemas differ")

    base_steps, aug_steps = len(base_data), len(aug_data)
    base_motion_ids = base_data.motion_id
    new_id_start = int(base_motion_ids.max().item()) + 1
    new_data_motion_ids = aug_data.motion_id.clone().to(torch.int64) + new_id_start
    fields = list(base_data.__dataclass_fields__)
    if fields != list(aug_data.__dataclass_fields__):
        raise ValueError("Motion tensor fields differ")

    mm = {}
    for name in fields:
        source = getattr(base_data, name)
        extra = getattr(aug_data, name)
        if source.shape[1:] != extra.shape[1:]:
            raise ValueError(f"Tensor shape mismatch for {name}: {source.shape} vs {extra.shape}")
        mm[name] = MemoryMappedTensor.empty(
            base_steps + aug_steps, *source.shape[1:], dtype=source.dtype
        )
        for lo in range(0, base_steps, 250_000):
            hi = min(base_steps, lo + 250_000)
            mm[name][lo:hi].copy_(source[lo:hi])
        for lo in range(0, aug_steps, 250_000):
            hi = min(aug_steps, lo + 250_000)
            data = new_data_motion_ids[lo:hi] if name == "motion_id" else extra[lo:hi]
            mm[name][base_steps + lo:base_steps + hi].copy_(data.to(dtype=source.dtype))
    combined = MotionData(**mm, batch_size=[base_steps + aug_steps])
    combined.memmap(str(ds_out), copy_existing=True)
    # MotionData.memmap records the tensorclass itself. MotionData.load expects
    # the nested payload to be a plain TensorDict and then wraps it as MotionData.
    tensor_meta = read_json(base / "dataset" / "_tensordict" / "meta.json")
    for value in tensor_meta.values():
        if isinstance(value, dict) and "shape" in value and value["shape"]:
            value["shape"][0] = base_steps + aug_steps
    tensor_meta["shape"] = [base_steps + aug_steps]
    tensor_meta["_type"] = "<class 'tensordict._td.TensorDict'>"
    (ds_out / "_tensordict" / "meta.json").write_text(json.dumps(tensor_meta))
    shutil.copy2(base / "dataset" / "meta.json", ds_out / "meta.json")

    starts = list(base_meta["starts"])
    ends = list(base_meta["ends"])
    starts.extend(base_steps + int(s) for s in aug_meta["starts"])
    ends.extend(base_steps + int(e) for e in aug_meta["ends"])
    meta = dict(base_meta)
    meta.update(starts=starts, ends=ends, info={})
    (ds_out / "meta_motion.json").write_text(json.dumps(meta))

    base_slots = np.load(base / "slots.npy", mmap_mode="r")
    base_forces = np.load(base / "forces.npy", mmap_mode="r")
    aug_slots = np.load(aug / "slots.npy", mmap_mode="r")
    aug_forces = np.load(aug / "forces.npy", mmap_mode="r")
    slots = np.lib.format.open_memmap(out / "slots.npy", mode="w+", dtype=np.int16,
                                     shape=(len(base_slots) + len(aug_slots), 3))
    forces = np.lib.format.open_memmap(out / "forces.npy", mode="w+", dtype=np.float32,
                                       shape=(len(base_forces) + len(aug_forces), 3, 3))
    slots[:len(base_slots)] = base_slots
    slots[len(base_slots):] = aug_slots
    forces[:len(base_forces)] = base_forces
    forces[len(base_forces):] = aug_forces
    slots.flush(); forces.flush()

    base_offsets = np.load(base / "motion_offsets.npy")
    aug_offsets = np.load(aug / "motion_offsets.npy")
    offsets = np.full(new_id_start + len(aug_offsets), -1, dtype=np.int64)
    offsets[:len(base_offsets)] = base_offsets
    offsets[new_id_start:new_id_start + len(aug_offsets)] = len(base_forces) + aug_offsets
    np.save(out / "motion_offsets.npy", offsets)

    # Verify every trainable span resolves to a valid force interval.
    checks = []
    for span_start, span_end in zip(starts, ends):
        mid = int(combined.motion_id[span_start].item())
        step0 = int(combined.step[span_start].item())
        step1 = int(combined.step[span_end - 1].item())
        if mid >= len(offsets) or offsets[mid] < 0 or offsets[mid] + step1 >= len(forces):
            raise RuntimeError(f"Force cache index does not cover span {mid}: {span_start}:{span_end}")
        checks.append((mid, step1))

    reports = list(base_manifest["reports"])
    for r in aug_manifest["reports"]:
        rr = dict(r)
        rr["motion_id"] = new_id_start + int(r["motion_id"])
        rr["start"] = base_steps + int(aug_meta["starts"][int(r["motion_id"])])
        rr["end"] = base_steps + int(aug_meta["ends"][int(r["motion_id"])])
        reports.append(rr)
    manifest = dict(base_manifest)
    manifest.update(
        split="train_augmented_single_hand",
        clips=int(base_manifest["clips"]) + int(aug_manifest["clips"]),
        frames=int(base_manifest["frames"]) + int(aug_manifest["frames"]),
        accepted_frames=int(base_manifest["accepted_frames"]) + int(aug_manifest["accepted_frames"]),
        accepted_spans=len(starts),
        point_frames=[int(a) + int(b) for a, b in zip(base_manifest["point_frames"], aug_manifest["point_frames"])],
        reports=reports,
        augmentation=dict(source=str(aug), added_clips=int(aug_manifest["clips"]),
                          added_frames=int(aug_manifest["frames"]), added_motion_id_start=new_id_start,
                          raw_base_steps=base_steps, indexed_spans_checked=len(checks)),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = dict(
        base_cache=str(base), augmentation_cache=str(aug), output=str(out),
        base_data_steps=base_steps, added_data_steps=aug_steps,
        base_motion_id_max=int(base_motion_ids.max().item()), added_motion_id_start=new_id_start,
        combined_motion_ids=len(offsets), combined_train_spans=len(starts),
        force_frames=len(forces), added_clips=int(aug_manifest["clips"]),
        added_left_clips=sum(r["side"] == "left" for r in aug_manifest["reports"]),
        added_right_clips=sum(r["side"] == "right" for r in aug_manifest["reports"]),
        all_spans_resolve=True,
    )
    (out / "merge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
