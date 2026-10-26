#!/usr/bin/env python3
"""Audit accepted CEER segmented training spans and write a hard-linked screened view."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from active_adaptation.utils.motion import MotionData


def source_for(mid: int) -> str:
    if mid < 8082:
        return "base_train18"
    if mid < 12291:
        return "single_hand_force"
    return "static_zero_force"


def joint_velocity_limits(names: list[str]) -> np.ndarray:
    """Configured G1 velocity_limit_sim values from active_adaptation/assets/humanoid.py."""
    limits = []
    for name in names:
        if any(token in name for token in ("hip_roll_joint", "knee_joint")):
            limits.append(20.0)
        elif name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
            limits.append(22.0)
        elif any(token in name for token in ("hip_yaw_joint", "hip_pitch_joint", "waist_yaw_joint")):
            limits.append(32.0)
        else:
            limits.append(37.0)
    return np.asarray(limits, dtype=np.float32)


def hardlink_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for current, dirs, files in os.walk(src):
        rel = Path(current).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(exist_ok=True)
        for d in dirs:
            (target_dir / d).mkdir(exist_ok=True)
        for name in files:
            rel_file = rel / name
            if str(rel_file) in {"manifest.json", "dataset/meta_motion.json"}:
                continue
            os.link(Path(current) / name, target_dir / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-root-speed", type=float, default=3.0)
    p.add_argument("--min-root-height", type=float, default=0.55)
    p.add_argument("--max-tilt", type=float, default=1.0)
    p.add_argument("--joint-speed-ratio", type=float, default=1.2)
    p.add_argument("--joint-speed-frame-fraction", type=float, default=0.05)
    p.add_argument("--min-span-frames", type=int, default=50)
    args = p.parse_args()
    src, out = args.input.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)

    manifest = json.loads((src / "manifest.json").read_text())
    meta_path = src / "dataset/meta_motion.json"
    meta = json.loads(meta_path.read_text())
    data = MotionData.load(str(src / "dataset"))
    slots = np.load(src / "slots.npy", mmap_mode="r")
    forces = np.load(src / "forces.npy", mmap_mode="r")
    offsets = np.load(src / "motion_offsets.npy", mmap_mode="r")
    num_points = len(manifest["points"])
    report_by_mid = {int(r.get("motion_id", -1)): r for r in manifest["reports"]}
    vel_limits = joint_velocity_limits(meta["joint_names"])
    keep_starts, keep_ends, removed = [], [], []
    source_summary = {}
    maxima = dict(force_point_N=0., force_total_N=0., force_slew_N_s=0.,
                  root_speed_m_s=0., joint_speed_ratio=0., root_tilt_rad=0.)

    for span_index, (start, end) in enumerate(zip(meta["starts"], meta["ends"])):
        start, end = int(start), int(end)
        mid = int(data.motion_id[start].item())
        frame_step = data.step[start:end].numpy()
        reasons = []
        if end - start < args.min_span_frames:
            reasons.append("span_shorter_than_minimum")
        if not np.all(data.motion_id[start:end].numpy() == mid):
            reasons.append("motion_id_changes_inside_span")
        if not np.array_equal(frame_step, np.arange(frame_step[0], frame_step[0] + len(frame_step))):
            reasons.append("nonconsecutive_motion_steps")
        if mid >= len(offsets) or int(offsets[mid]) < 0:
            reasons.append("missing_force_offset")

        root = data.root_pos_w[start:end].numpy().astype(np.float32)
        quat = data.root_quat_w[start:end].numpy().astype(np.float32)
        qnorm = np.linalg.norm(quat, axis=-1)
        qsafe = quat / np.maximum(qnorm[:, None], 1.e-8)
        tilt = np.arccos(np.clip(1. - 2. * (qsafe[:, 1] ** 2 + qsafe[:, 2] ** 2), -1., 1.))
        root_speed = np.linalg.norm(data.root_lin_vel_w[start:end, :2].numpy().astype(np.float32), axis=-1)
        jvel = np.abs(data.joint_vel[start:end].numpy().astype(np.float32))
        jpos = data.joint_pos[start:end].numpy().astype(np.float32)
        bodies = data.body_pos_w[start:end].numpy().astype(np.float32)
        for arr in (root, quat, jvel, jpos, bodies):
            if not np.isfinite(arr).all():
                reasons.append("nonfinite_motion_state")
                break
        if float(qnorm.min()) < .98 or float(qnorm.max()) > 1.02:
            reasons.append("invalid_root_quaternion_norm")
        if float(root[:, 2].min()) < args.min_root_height:
            reasons.append("root_height_below_threshold")
        if float(tilt.max()) >= args.max_tilt:
            reasons.append("root_tilt_above_threshold")
        if float(root_speed.max(initial=0.)) > args.max_root_speed:
            reasons.append("root_horizontal_speed_above_threshold")
        speed_ratio = jvel / vel_limits[None, :]
        if float((speed_ratio > args.joint_speed_ratio).any(axis=1).mean()) > args.joint_speed_frame_fraction:
            reasons.append("joint_speed_exceeds_actuator_margin_sustained")
        if float(np.abs(jpos).max(initial=0.)) >= np.pi:
            reasons.append("joint_position_outside_generic_range")

        s = f = None
        if mid < len(offsets) and int(offsets[mid]) >= 0:
            f0 = int(offsets[mid]) + int(frame_step[0])
            f1 = int(offsets[mid]) + int(frame_step[-1]) + 1
            s = np.asarray(slots[f0:f1])
            f = np.asarray(forces[f0:f1])
            if len(s) != len(frame_step) or len(f) != len(frame_step):
                reasons.append("force_coverage_mismatch")
            else:
                if not np.isfinite(f).all():
                    reasons.append("nonfinite_force")
                if np.any((s < -1) | (s >= num_points)):
                    reasons.append("force_slot_out_of_range")
                if np.any(np.abs(f[s < 0]) > 1.e-6):
                    reasons.append("force_nonzero_in_disabled_slot")
                dense = np.zeros((len(s), num_points, 3), dtype=np.float32)
                duplicate_slot = any(len(set(row[row >= 0].tolist())) != int((row >= 0).sum()) for row in s)
                for k in range(s.shape[1]):
                    active = s[:, k] >= 0
                    rows = np.where(active)[0]
                    ids = s[active, k]
                    if len(rows):
                        dense[rows, ids] += f[active, k]
                if duplicate_slot:
                    reasons.append("duplicate_point_slots")
                point_norm = np.linalg.norm(dense, axis=-1)
                point_max = float(point_norm.max(initial=0.))
                total_max = float(point_norm.sum(axis=-1).max(initial=0.))
                slew = float(np.linalg.norm(np.diff(dense, axis=0), axis=-1).max(initial=0.) / .02)
                maxima["force_point_N"] = max(maxima["force_point_N"], point_max)
                maxima["force_total_N"] = max(maxima["force_total_N"], total_max)
                maxima["force_slew_N_s"] = max(maxima["force_slew_N_s"], slew)
                if point_max > 30.001:
                    reasons.append("per_point_force_above_30N")
                if total_max > 60.001:
                    reasons.append("total_force_above_60N")
                if slew > 150.01:
                    reasons.append("force_slew_above_150N_s")
                changed = np.where(np.any(s[1:] != s[:-1], axis=1))[0]
                if len(changed) and np.any(dense[changed] > 1.e-6):
                    reasons.append("force_slot_switch_without_zero_transition")

        report_record = report_by_mid.get(mid)
        if report_record and report_record.get("profile") == "static hold, zero force":
            static_ok = (np.max(np.abs(np.diff(root, axis=0)), initial=0.) <= 1.e-6 and
                         np.max(np.abs(np.diff(qsafe, axis=0)), initial=0.) <= 1.e-6 and
                         np.max(np.abs(np.diff(jpos, axis=0)), initial=0.) <= 1.e-6 and
                         np.max(np.abs(np.diff(bodies, axis=0)), initial=0.) <= 1.e-6 and
                         np.max(jvel, initial=0.) <= 1.e-6)
            if not static_ok:
                reasons.append("static_hold_not_actually_static")
            if s is None or f is None or np.any(s >= 0) or np.max(np.abs(f), initial=0.) > 1.e-6:
                reasons.append("static_hold_has_force")

        maxima["root_speed_m_s"] = max(maxima["root_speed_m_s"], float(root_speed.max(initial=0.)))
        maxima["joint_speed_ratio"] = max(maxima["joint_speed_ratio"], float(speed_ratio.max(initial=0.)))
        maxima["root_tilt_rad"] = max(maxima["root_tilt_rad"], float(tilt.max(initial=0.)))
        group = source_for(mid)
        g = source_summary.setdefault(group, dict(spans=0, frames=0, removed_spans=0, removed_frames=0))
        g["spans"] += 1; g["frames"] += end - start
        if reasons:
            removed.append(dict(span_index=span_index, motion_id=mid, start=start, end=end,
                                frames=end-start, reasons=sorted(set(reasons)),
                                max_root_speed_m_s=float(root_speed.max(initial=0.))))
            g["removed_spans"] += 1; g["removed_frames"] += end-start
        else:
            keep_starts.append(start); keep_ends.append(end)

    if not keep_starts:
        raise RuntimeError("Screening would remove every training span")
    hardlink_tree(src, out)
    screened_meta = dict(meta, starts=keep_starts, ends=keep_ends)
    (out / "dataset/meta_motion.json").write_text(json.dumps(screened_meta))
    screened_manifest = dict(manifest)
    screened_manifest.update(split="train_screened", accepted_spans=len(keep_starts),
        accepted_frames=int(sum(e-s for s, e in zip(keep_starts, keep_ends))))
    screened_manifest["quality_screen"] = dict(
        criteria=dict(min_span_frames=args.min_span_frames, root_height_min_m=args.min_root_height,
            root_tilt_max_rad=args.max_tilt, root_horizontal_speed_max_m_s=args.max_root_speed,
            joint_speed_max_ratio=args.joint_speed_ratio,
            joint_speed_exceedance_frame_fraction_max=args.joint_speed_frame_fraction,
            force_per_point_max_N=30., force_total_max_N=60., force_slew_max_N_s=150.),
        input_spans=len(meta["starts"]), kept_spans=len(keep_starts), removed_spans=len(removed),
        input_trainable_frames=int(sum(e-s for s, e in zip(meta["starts"], meta["ends"]))),
        kept_trainable_frames=int(sum(e-s for s, e in zip(keep_starts, keep_ends))),
        removed=removed)
    (out / "manifest.json").write_text(json.dumps(screened_manifest, indent=2))
    audit = dict(input=str(src), output=str(out), criteria=screened_manifest["quality_screen"]["criteria"],
        source_summary=source_summary, maxima=maxima, kept_spans=len(keep_starts), removed=removed,
        kept_frames=screened_manifest["accepted_frames"],
        original_frames=len(data), screening_is_non_destructive=True)
    (out / "screening_report.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "removed"}, indent=2), flush=True)
    if removed:
        print("REMOVED_SPANS", json.dumps(removed, indent=2), flush=True)


if __name__ == "__main__":
    main()
