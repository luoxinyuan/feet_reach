#!/usr/bin/env python3
"""Export the fixed feet_reach tracking protocol clips into SuperMimic's NPZ format."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Import just the motion-data module; the package __init__ imports the full RL
# stack, which is unrelated to exporting protocol clips and may need torchrl.
active_pkg = types.ModuleType("active_adaptation")
active_pkg.__path__ = [str(ROOT / "active_adaptation")]
utils_pkg = types.ModuleType("active_adaptation.utils")
utils_pkg.__path__ = [str(ROOT / "active_adaptation" / "utils")]
sys.modules.setdefault("active_adaptation", active_pkg)
sys.modules.setdefault("active_adaptation.utils", utils_pkg)
from active_adaptation.utils.motion import MotionDataset, G1_JOINT_NAMES

manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/compliance_eval/protocol_v1.json"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts/compliance_eval/20261108-supermimic-e11c-benchmark/protocol_motions.npz"
manifest = json.loads(manifest_path.read_text())
mem_root = Path(os.environ.get("MEMPATH", ROOT / "dataset"))
datasets = {}
qpos_parts, qvel_parts, lengths, names, rows = [], [], [], [], []
starts = []
cursor = 0

for row in manifest["clips"]:
    rel = row["dataset"]
    if rel not in datasets:
        datasets[rel] = MotionDataset.create_from_path_lazy(rel, device="cpu")
    ds = datasets[rel]
    clip = int(row["clip"])
    # The protocol's clip index already points at the selected segment in the
    # dataset. source_start_step is provenance relative to the original source
    # motion; adding it again would shift/truncate segmented training clips.
    start = int(ds.starts[clip])
    length = min(int(row["frames"]), int(ds.ends[clip]) - start)
    if length < 2:
        raise ValueError(f"Protocol row has fewer than two frames: {row}")
    joint_perm = [ds.joint_names.index(name) for name in G1_JOINT_NAMES]
    sl = slice(start, start + length)
    root_pos = ds.data.root_pos_w[sl].to(torch.float32).cpu().numpy()
    root_quat = ds.data.root_quat_w[sl].to(torch.float32).cpu().numpy()
    joint_pos = ds.data.joint_pos[sl][:, joint_perm].to(torch.float32).cpu().numpy()
    root_lin = ds.data.root_lin_vel_w[sl].to(torch.float32).cpu().numpy()
    root_ang = ds.data.root_ang_vel_w[sl].to(torch.float32).cpu().numpy()
    joint_vel = ds.data.joint_vel[sl][:, joint_perm].to(torch.float32).cpu().numpy()
    qpos_parts.append(np.concatenate([root_pos, root_quat, joint_pos], axis=-1))
    qvel_parts.append(np.concatenate([root_lin, root_ang, joint_vel], axis=-1))
    starts.append(cursor)
    lengths.append(length)
    cursor += length
    names.append(f"{row['split']}/{rel}/{clip}:{row['category']}")
    rows.append({**row, "frames_exported": length})

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_path,
    qpos=np.concatenate(qpos_parts).astype(np.float32),
    qvel=np.concatenate(qvel_parts).astype(np.float32),
    starts=np.asarray(starts, dtype=np.int64),
    lengths=np.asarray(lengths, dtype=np.int64),
    names=np.asarray(names),
    fps=np.asarray(50.0, dtype=np.float32),
)
(out_path.with_suffix(".json")).write_text(json.dumps({
    "source_manifest": str(manifest_path.resolve()),
    "manifest_version": manifest["version"],
    "motion_count": len(rows),
    "frame_count": int(sum(lengths)),
    "fps": 50,
    "joint_names": G1_JOINT_NAMES,
    "clips": rows,
}, indent=2))
print(json.dumps({"motion_count": len(rows), "frame_count": int(sum(lengths)), "path": str(out_path)}), flush=True)
