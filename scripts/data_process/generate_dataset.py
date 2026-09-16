import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.motion import MotionDataset


def preprocess_motion(motion, foot_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]  # (T,3)
    offset_xy = root_pos[0, :2].copy()  # x and y coordinates of the first frame
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    z_l = motion["xpos"][:, foot_idx[0], 2]
    z_r = motion["xpos"][:, foot_idx[1], 2]

    if not always_on_ground:
        z_min = float(min(z_l.min(), z_r.min()))
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz
        motion["xpos"][:, :, 2] += dz
    else:
        z_min = np.min(
            np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
            axis=-1,
            keepdims=True,
        )
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz.reshape(-1)
        motion["xpos"][:, :, 2] += dz
    return motion


def none_callback(_ctx, m):
    m["metadata"] = None


def _default_relpath(dataset_root: Path, p: Path) -> str:
    if dataset_root.is_file():
        return p.name
    try:
        return str(p.relative_to(dataset_root))
    except ValueError:
        return str(p)


def load_allowlist(path: str) -> set[tuple[str, int, int]]:
    payload = json.loads(Path(path).read_text())
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("allowlist json missing required object key: params")
    segs = payload.get("segments", {})
    out: set[tuple[str, int, int]] = set()
    for fname, spans in segs.items():
        for start, end in spans:
            out.add((str(fname), int(start), int(end)))
    required_params = ["target_fps", "pad_before", "pad_after", "segment_len"]
    missing = [k for k in required_params if k not in params]
    if missing:
        raise ValueError(f"allowlist json missing required params keys: {missing}")
    return out, {
        "target_fps": int(params["target_fps"]),
        "pad_before": int(params["pad_before"]),
        "pad_after": int(params["pad_after"]),
        "segment_len": int(params["segment_len"]),
    }


def make_allowlist_filter(allow: set[tuple[str, int, int]], dataset_root: Path):
    def _filter(_motion, _foot_idx, p, start_idx, end_idx):
        rel = _default_relpath(dataset_root, Path(p))
        return (rel, int(start_idx), int(end_idx)) in allow

    return _filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="NPZ file or directory to convert")
    ap.add_argument("--allowlist", default=None, help="Optional allowlist json from precompute step")
    ap.add_argument("--mem-path", required=True, help="Output memmap directory")
    ap.add_argument("--target-fps", type=int, default=50)
    ap.add_argument("--pad-before", type=int, default=25)
    ap.add_argument("--pad-after", type=int, default=50)
    ap.add_argument("--segment-len", type=int, default=1000)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    params = {
        "target_fps": args.target_fps,
        "pad_before": args.pad_before,
        "pad_after": args.pad_after,
        "segment_len": args.segment_len,
    }
    allow_filter = None
    if args.allowlist is not None:
        allow, params = load_allowlist(args.allowlist)
        allow_filter = make_allowlist_filter(allow, dataset_root)

    MotionDataset.create_from_path(
        str(dataset_root),
        target_fps=params["target_fps"],
        mem_path=args.mem_path,
        callback=none_callback,
        motion_processer=partial(preprocess_motion, always_on_ground=True),
        motion_filter=allow_filter,
        pad_before=params["pad_before"],
        pad_after=params["pad_after"],
        segment_len=params["segment_len"],
        storage_float_dtype=torch.float16,
        storage_int_dtype=torch.int32,
    )


if __name__ == "__main__":
    main()
