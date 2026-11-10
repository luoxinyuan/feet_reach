"""Build a small CEER/MotionDataset-compatible crouch augmentation set.

Sources:
* local AMASS G1 MotionData, screened for sustained bilateral knee flexion and
  low root height while upright, with short standing context around transitions;
* non-mirrored, flat-ground-retained BONES-SEED G1 clips labeled as free-standing
  crouch-idle or squat (wall-supported and locomotion clips are excluded).

The output is separate from the existing full AMASS dataset. Mix it at the
dataset-path level with a small path weight (recommended 0.05 against 1.0 for
the original AMASS path); the loader samples paths before clips, so equal path
weights would over-sample this small augmentation dramatically.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import MemoryMappedTensor

ROOT = Path(__file__).resolve().parents[2]
AMASS = Path("/home/xl521/hiearchical_compliance/dataset/amass")
BONES_META = Path("/home/xl521/supermimic_force_transition/data/raw/bones_seed/metadata/seed_metadata_v004.csv")
BONES_CSV = Path("/home/xl521/supermimic_force_transition/data/raw/bones_seed/g1/csv")
BONES_FILTERED = Path("/home/xl521/supermimic_force_transition/data/processed/bones_seed_filtered_50hz.npz")
MJCF = ROOT / "deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml"
OUT = ROOT / "dataset/crouch_lowlevel"
FPS = 50
FOOT_Z = 0.03540915

sys.path.insert(0, str(ROOT))
from active_adaptation.utils.motion import MotionData  # noqa: E402

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def quat_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate((q_wxyz[..., 1:], q_wxyz[..., :1]), axis=-1)


def stable_split(key: str, val_mod: int = 5) -> str:
    n = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return "val" if n % val_mod == 0 else "train"


def amass_crouch_windows(meta: dict) -> tuple[list[dict], dict]:
    """Find restrained, upright crouch-transition windows in local processed AMASS."""
    total = int(meta["ends"][-1])
    root_pos = np.memmap(AMASS / "_tensordict/root_pos_w.memmap", dtype=np.float16, mode="r", shape=(total, 3))
    root_quat = np.memmap(AMASS / "_tensordict/root_quat_w.memmap", dtype=np.float16, mode="r", shape=(total, 4))
    root_vel = np.memmap(AMASS / "_tensordict/root_lin_vel_w.memmap", dtype=np.float16, mode="r", shape=(total, 3))
    joint_pos = np.memmap(AMASS / "_tensordict/joint_pos.memmap", dtype=np.float16, mode="r", shape=(total, 29))
    k_l, k_r = (meta["joint_names"].index(n) for n in ("left_knee_joint", "right_knee_joint"))

    # A candidate low-crouch frame is bilateral knee flexion > 0.65 rad,
    # pelvis height < 0.70 m, and upright root. Require a 0.30 s continuous run,
    # low/moderate travel speed, and a nearby higher standing posture.
    knee_threshold, height_threshold, min_run = 0.65, 0.70, 15
    context = 50
    candidates: list[dict] = []
    crouch_frames = 0
    for clip_id, (s, e) in enumerate(zip(meta["starts"], meta["ends"])):
        s, e = int(s), int(e)
        z = np.asarray(root_pos[s:e, 2], dtype=np.float32)
        q = np.asarray(root_quat[s:e], dtype=np.float32)
        kp = np.asarray(joint_pos[s:e, [k_l, k_r]], dtype=np.float32).mean(axis=-1)
        speed = np.linalg.norm(np.asarray(root_vel[s:e, :2], dtype=np.float32), axis=-1)
        upright = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2) > 0.90
        mask = (kp > knee_threshold) & (z < height_threshold) & upright
        where = np.flatnonzero(mask)
        if not where.size:
            continue
        runs: list[tuple[int, int]] = []
        a = prev = int(where[0])
        for x in where[1:]:
            x = int(x)
            if x != prev + 1:
                if prev - a + 1 >= min_run:
                    runs.append((a, prev + 1))
                a = x
            prev = x
        if prev - a + 1 >= min_run:
            runs.append((a, prev + 1))

        accepted: list[tuple[int, int, int, int]] = []
        for a, b in runs:
            if float(np.percentile(speed[a:b], 75)) > 0.65:
                continue
            standing_before = a >= 5 and float(np.median(z[max(0, a - 25):a])) > height_threshold + 0.03
            standing_after = b < len(z) - 5 and float(np.median(z[b:min(len(z), b + 25)])) > height_threshold + 0.03
            if not (standing_before or standing_after):
                continue
            accepted.append((max(0, a - context), min(len(z), b + context), a, b))

        if not accepted:
            continue
        accepted.sort()
        merged: list[tuple[int, int, int, int]] = []
        for a, b, ca, cb in accepted:
            if merged and a <= merged[-1][1]:
                old_a, old_b, old_ca, old_cb = merged[-1]
                merged[-1] = (old_a, max(old_b, b), min(old_ca, ca), max(old_cb, cb))
            else:
                merged.append((a, b, ca, cb))
        for a, b, ca, cb in merged:
            # Keep windows compact and within the same clip. No segment exceeds 8 s.
            if b - a > 400:
                center = (ca + cb) // 2
                a, b = max(0, center - 175), min(e - s, center + 175)
            if b - a < 60:
                continue
            candidates.append({
                "source": "amass_kinematic",
                "source_clip": clip_id,
                "start": a,
                "end": b,
                "crouch_start": ca,
                "crouch_end": cb,
                "split": stable_split(f"amass:{clip_id}"),
                "frames": b - a,
            })
            crouch_frames += cb - ca

    return candidates, {
        "source": str(AMASS), "source_clips": len(meta["starts"]),
        "screen": {"mean_bilateral_knee_flex_rad_gt": knee_threshold,
                   "root_height_m_lt": height_threshold, "root_upright_dot_gt": 0.90,
                   "minimum_run_frames": min_run, "context_each_side_frames": context,
                   "p75_horizontal_speed_m_s_max": 0.65},
        "selected_windows": len(candidates),
        "selected_frames_with_context": sum(x["frames"] for x in candidates),
        "crouch_core_frames_before_context": crouch_frames,
    }


def bones_rows() -> tuple[list[dict], dict]:
    with zipfile.ZipFile(BONES_FILTERED) as zf:
        retained = set(np.load(zf.open("names.npy"), allow_pickle=True).astype(str).tolist())
        lengths = np.load(zf.open("lengths.npy"), allow_pickle=True)
        names = np.load(zf.open("names.npy"), allow_pickle=True).astype(str).tolist()
    retained_lengths = dict(zip(names, map(int, lengths)))

    selected: list[dict] = []
    counts: Counter[str] = Counter()
    with BONES_META.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["is_mirror"].lower() == "true":
                continue
            name = row["content_name"].lower()
            if not (name == "squat" or name.startswith("crouch_idle")):
                continue
            if any(x in name for x in ("_jog", "_walk", "_run", "_crawl", "_turn")):
                continue
            rel = row["move_g1_path"].removeprefix("g1/csv/")
            if rel not in retained or retained_lengths.get(rel, 0) < 60:
                continue
            # Exclude any props/support or non-static combined action even if its
            # semantic name contains crouch. Only free squat/crouch holds/transitions.
            text = " ".join(row.get(k, "") for k in (
                "content_natural_desc_1", "content_natural_desc_2",
                "content_natural_desc_3", "content_natural_desc_4",
                "content_technical_description", "content_props",
            )).lower()
            if any(x in text for x in ("wall", "cupboard", "chair", "crawl", "burpee", "lying", "kneel", "jog", "walking", "running")):
                continue
            path = BONES_CSV / rel
            if not path.is_file():
                continue
            selected.append({
                "source": "bones_seed", "source_name": rel,
                "content_name": row["content_name"], "actor": row["take_actor"],
                "frames_expected": retained_lengths[rel],
                "split": stable_split(f"bones-actor:{row['take_actor']}"),
            })
            counts[row["content_name"]] += 1
    if not selected:
        raise RuntimeError("No eligible BONES-SEED squat/crouch clips found")
    return selected, {
        "source": str(BONES_CSV), "metadata": str(BONES_META),
        "eligibility": "non-mirror, retained by BONES flat-ground filter, free-standing squat or crouch_idle labels",
        "selected_clips": len(selected),
        "selected_frames_expected": sum(x["frames_expected"] for x in selected),
        "categories": dict(counts),
        "split": "by take_actor hash; actor leakage avoided between train and val",
    }


def marker_transforms(amass_data: MotionData, meta: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Estimate fixed head/hand marker transforms from the matching AMASS G1 schema."""
    transforms = {}
    pairs = {
        "head_mimic": ("torso_link", "head_mimic"),
        "left_hand_mimic": ("left_wrist_roll_link", "left_hand_mimic"),
        "right_hand_mimic": ("right_wrist_roll_link", "right_hand_mimic"),
    }
    data = amass_data
    for marker, (parent, child) in pairs.items():
        pi, ci = meta["body_names"].index(parent), meta["body_names"].index(child)
        start = int(meta["starts"][0]); end = min(int(meta["ends"][0]), start + 1000)
        pp = data.body_pos_w[start:end, pi].float().numpy()
        cp = data.body_pos_w[start:end, ci].float().numpy()
        pq = data.body_quat_w[start:end, pi].float().numpy()
        cq = data.body_quat_w[start:end, ci].float().numpy()
        rp = Rotation.from_quat(quat_xyzw(pq))
        rc = Rotation.from_quat(quat_xyzw(cq))
        local_pos = rp.inv().apply(cp - pp)
        local_rot = rp.inv() * rc
        pos = np.median(local_pos, axis=0)
        rot = local_rot.mean().as_quat()  # xyzw for XML conversion below
        transforms[marker] = (pos.astype(float), rot.astype(float))
    return transforms


def model_with_markers(transforms: dict[str, tuple[np.ndarray, np.ndarray]]) -> mujoco.MjModel:
    tree = ET.parse(MJCF)
    root = tree.getroot()
    for marker, parent in (("head_mimic", "torso_link"),
                           ("left_hand_mimic", "left_wrist_roll_link"),
                           ("right_hand_mimic", "right_wrist_roll_link")):
        pos, qxyzw = transforms[marker]
        quat_wxyz = [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]
        parent_node = root.find(f".//body[@name='{parent}']")
        if parent_node is None:
            raise RuntimeError(f"MJCF parent body missing: {parent}")
        ET.SubElement(parent_node, "body", {
            "name": marker,
            "pos": " ".join(f"{x:.9g}" for x in pos),
            "quat": " ".join(f"{x:.9g}" for x in quat_wxyz),
        })
    # Keep the temporary XML beside its source so relative mesh paths resolve.
    # Meshes are only visual; the FK model removes them below before compiling.
    temp = MJCF.parent / "_crouch_fk_model.xml"
    tree.write(temp)
    model = mujoco.MjModel.from_xml_path(str(temp))
    temp.unlink(missing_ok=True)
    required = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
                "head_mimic", "left_hand_mimic", "right_hand_mimic"]
    for name in required:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise RuntimeError(f"Generated FK model missing {name}")
    if model.nq != 36:
        raise RuntimeError(f"FK model expected 36 qpos, got {model.nq}")
    return model


def load_bones_converter():
    code_root = Path("/home/xl521/supermimic_force_transition/code")
    sys.path.insert(0, str(code_root))
    path = code_root / "scripts/prepare_motions.py"
    spec = importlib.util.spec_from_file_location("bones_prepare_motions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def fk_bones_clip(row: dict, model: mujoco.MjModel, converter) -> dict[str, np.ndarray]:
    pos, quat, dof = converter.load_bones_csv(str(BONES_CSV / row["source_name"]))
    pos, quat, dof = converter.resample(pos, quat, dof, 120.0, FPS)
    lin, ang, dof_vel = converter.velocities(pos, quat, dof, 1.0 / FPS)
    pos = pos.numpy().astype(np.float32)
    quat = quat.numpy().astype(np.float32)  # wxyz
    dof = dof.numpy().astype(np.float32)
    qvel = torch.cat((lin, ang, dof_vel), dim=-1).numpy().astype(np.float32)

    dof_order = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    if any(j < 0 for j in dof_order):
        raise RuntimeError("FK model is missing a required G1 joint")
    qadr = [int(model.jnt_qposadr[j]) for j in dof_order]
    if dof.shape[1] != len(qadr):
        raise RuntimeError(f"BONES joint count mismatch: {dof.shape[1]} vs {len(qadr)}")

    wanted = [
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
        "right_elbow_link", "right_wrist_roll_link", "head_mimic",
        "left_hand_mimic", "right_hand_mimic",
    ]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in wanted]
    if any(b < 0 for b in body_ids):
        raise RuntimeError("FK model does not expose all MotionDataset body names")
    data = mujoco.MjData(model)
    qpos = np.empty((len(pos), 36), dtype=np.float32)
    body_pos = np.empty((len(pos), len(wanted), 3), dtype=np.float32)
    body_quat = np.empty((len(pos), len(wanted), 4), dtype=np.float32)
    for t in range(len(pos)):
        qpos[t, :3] = pos[t]
        qpos[t, 3:7] = quat[t]
        qpos[t, 7:] = dof[t]
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[body_ids]
        body_quat[t] = data.xquat[body_ids]

    # Match the AMASS preprocessing convention: put the lowest ankle frame at
    # the shared 3.54 cm ankle-origin height using one constant clip offset.
    feet = [wanted.index("left_ankle_roll_link"), wanted.index("right_ankle_roll_link")]
    dz = FOOT_Z - float(body_pos[:, feet, 2].min())
    qpos[:, 2] += dz
    body_pos[:, :, 2] += dz
    offset_xy = qpos[0, :2].copy()
    qpos[:, :2] -= offset_xy
    body_pos[:, :, :2] -= offset_xy[None, None, :]
    body_pos_b = np.empty_like(body_pos)
    for t in range(len(qpos)):
        rot = Rotation.from_quat(quat_xyzw(qpos[t, 3:7][None]))
        body_pos_b[t] = rot.inv().apply(body_pos[t] - qpos[t, None, :3])

    return {
        "root_pos_w": qpos[:, :3], "root_quat_w": qpos[:, 3:7],
        "root_lin_vel_w": qvel[:, :3], "root_ang_vel_w": qvel[:, 3:6],
        "joint_pos": qpos[:, 7:], "joint_vel": qvel[:, 6:],
        "body_pos_w": body_pos, "body_pos_b": body_pos_b, "body_quat_w": body_quat,
    }


FIELDS = {
    "motion_id": (torch.int32, ()), "step": (torch.int32, ()),
    "root_pos_w": (torch.float16, (3,)), "root_quat_w": (torch.float16, (4,)),
    "root_lin_vel_w": (torch.float16, (3,)), "root_ang_vel_w": (torch.float16, (3,)),
    "joint_pos": (torch.float16, (29,)), "joint_vel": (torch.float16, (29,)),
    "body_pos_w": (torch.float16, (27, 3)), "body_pos_b": (torch.float16, (27, 3)),
    "body_quat_w": (torch.float16, (27, 4)),
}


def write_split(split: str, clips: list[dict], body_names: list[str], joint_names: list[str]) -> dict:
    dest = OUT / split
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dest}")
    dest.mkdir(parents=True)
    total_frames = sum(len(c["fields"]["root_pos_w"]) for c in clips)
    mm = {name: MemoryMappedTensor.empty(total_frames, *shape, dtype=dtype) for name, (dtype, shape) in FIELDS.items()}
    cursor, starts, ends = 0, [], []
    for motion_id, clip in enumerate(clips):
        fields = clip["fields"]
        n = len(fields["root_pos_w"])
        starts.append(cursor)
        ends.append(cursor + n)
        mm["motion_id"][cursor:cursor+n] = motion_id
        mm["step"][cursor:cursor+n] = torch.arange(n, dtype=torch.int32)
        for key in FIELDS:
            if key in ("motion_id", "step"):
                continue
            mm[key][cursor:cursor+n] = torch.as_tensor(fields[key], dtype=torch.float16)
        cursor += n
    data = MotionData(**mm, batch_size=[total_frames])
    data.memmap(str(dest))
    meta = {"body_names": body_names, "joint_names": joint_names,
            "starts": starts, "ends": ends, "info": {}}
    (dest / "meta_motion.json").write_text(json.dumps(meta))
    return {"clips": len(clips), "frames": total_frames,
            "hours_at_50hz": total_frames / FPS / 3600,
            "length_min": min((e-s for s, e in zip(starts, ends)), default=0),
            "length_max": max((e-s for s, e in zip(starts, ends)), default=0)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    amass_meta = json.loads((AMASS / "meta_motion.json").read_text())
    amass_data = MotionData.load(str(AMASS))
    amass_windows, amass_report = amass_crouch_windows(amass_meta)
    bones, bones_report = bones_rows()
    fk_model = model_with_markers(marker_transforms(amass_data, amass_meta))
    converter = load_bones_converter()

    selected_names = {c["source_name"] for c in bones}
    bones_cache: dict[str, dict[str, np.ndarray]] = {}
    for c in bones:
        if c["source_name"] not in bones_cache:
            bones_cache[c["source_name"]] = fk_bones_clip(c, fk_model, converter)
        c["fields"] = bones_cache[c["source_name"]]
        if len(c["fields"]["root_pos_w"]) != c["frames_expected"]:
            # The upstream filter and the common resampler should agree within one frame.
            delta = abs(len(c["fields"]["root_pos_w"]) - c["frames_expected"])
            if delta > 1:
                raise RuntimeError(f"BONES length changed unexpectedly for {c['source_name']}: {len(c['fields']['root_pos_w'])} vs {c['frames_expected']}")

    # Copy AMASS windows from the already-processed MotionData representation.
    for c in amass_windows:
        source_id = int(c["source_clip"])
        start = int(amass_meta["starts"][source_id]) + int(c["start"])
        end = int(amass_meta["starts"][source_id]) + int(c["end"])
        fields = {}
        xy = amass_data.root_pos_w[start, :2].float().clone()
        for key in FIELDS:
            if key in ("motion_id", "step"):
                continue
            value = getattr(amass_data, key)[start:end].float().clone()
            if key == "root_pos_w":
                value[:, :2] -= xy
            elif key == "body_pos_w":
                value[:, :, :2] -= xy[None, None, :]
            elif key in ("root_quat_w", "body_quat_w"):
                value /= value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            fields[key] = value
        c["fields"] = fields

    all_clips = amass_windows + bones
    clips_by_split = {s: [c for c in all_clips if c["split"] == s] for s in ("train", "val")}
    # Keep split order stable and make AMASS/BONES contributions interleaved.
    for split, clips in clips_by_split.items():
        if not any(c["source"] == "amass_kinematic" for c in clips):
            raise RuntimeError(f"No AMASS-derived {split} crouch windows")
        if not any(c["source"] == "bones_seed" for c in clips):
            raise RuntimeError(f"No BONES-SEED {split} crouch clips")
        clips.sort(key=lambda c: (c["source"], c.get("source_clip", c.get("source_name", "")), c.get("start", 0)))

    out_meta = {"body_names": amass_meta["body_names"], "joint_names": amass_meta["joint_names"]}
    report = {
        "created_by": "scripts/data_process/prepare_crouch_lowlevel_dataset.py",
        "fps": FPS,
        "schema": "active_adaptation.utils.motion.MotionData / CEER motion tracking",
        "sources": {"amass": amass_report, "bones_seed": bones_report},
        "split": {},
        "mixing_recommendation": {
            "base_path_weight": 1.0,
            "crouch_path_weight": 0.05,
            "effective_crouch_reset_probability": 0.05 / 1.05,
            "base_training_frames_reference": 4_286_066,
            "crouch_train_frame_fraction_of_base": None,
            "note": "Path weights are sampled per reset before selecting a clip; weights [1.0, 0.05] avoid 50/50 oversampling.",
        },
    }

    for split, clips in clips_by_split.items():
        per_source = Counter(c["source"] for c in clips)
        stats = write_split(split, clips, out_meta["body_names"], out_meta["joint_names"])
        stats["clips_by_source"] = dict(per_source)
        report["split"][split] = stats
    report["mixing_recommendation"]["crouch_train_frame_fraction_of_base"] = (
        report["split"]["train"]["frames"] / report["mixing_recommendation"]["base_training_frames_reference"]
    )
    # Keep source provenance out of tensor metadata but record every chosen interval.
    for split, clips in clips_by_split.items():
        (OUT / split / "source_manifest.json").write_text(json.dumps([
            {k: v for k, v in c.items() if k != "fields"} for c in clips
        ], indent=2))
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
