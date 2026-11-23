#!/usr/bin/env python3
"""Project motion datasets onto a shared body-keypoint schema, streaming to disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "torch.float16": np.float16,
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.uint8": np.uint8,
    "torch.bool": np.bool_,
}
BODY_FIELDS = {"body_pos_w", "body_pos_b", "body_quat_w"}


def project_dataset(source: Path, destination: Path, body_names: list[str], chunk_rows: int) -> None:
    source_meta_path = source / "_tensordict" / "meta.json"
    meta = json.loads(source_meta_path.read_text())
    source_motion = json.loads((source / "meta_motion.json").read_text())
    source_names = source_motion["body_names"]
    missing = [name for name in body_names if name not in source_names]
    if missing:
        raise ValueError(f"{source} is missing reference body names: {missing}")
    indices = np.asarray([source_names.index(name) for name in body_names], dtype=np.int64)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "_tensordict").mkdir()

    import shutil
    for sidecar in source.iterdir():
        if sidecar.is_file() and sidecar.name not in {"meta_motion.json"}:
            shutil.copyfile(sidecar, destination / sidecar.name)

    for field, info in meta.items():
        if field.startswith("_") or field in {"shape", "device", "_type"}:
            continue
        dtype = DTYPES.get(info.get("dtype"))
        if dtype is None:
            raise ValueError(f"Unsupported dtype for {field}: {info.get('dtype')}")
        old_shape = tuple(info["shape"])
        source_file = source / "_tensordict" / f"{field}.memmap"
        dest_file = destination / "_tensordict" / f"{field}.memmap"
        if field not in BODY_FIELDS:
            # Non-body tensors have identical row and feature schemas, so copy
            # their raw storage byte-for-byte.
            shutil.copyfile(source_file, dest_file)
            continue
        if old_shape[1] != len(source_names):
            raise ValueError(f"Unexpected body axis for {source}/{field}: {old_shape}")
        new_shape = (old_shape[0], len(body_names), *old_shape[2:])
        src = np.memmap(source_file, dtype=dtype, mode="r", shape=old_shape)
        dst = np.memmap(dest_file, dtype=dtype, mode="w+", shape=new_shape)
        for start in range(0, old_shape[0], chunk_rows):
            stop = min(start + chunk_rows, old_shape[0])
            dst[start:stop] = src[start:stop, indices]
        dst.flush()
        del src, dst
        info["shape"] = list(new_shape)

    (destination / "_tensordict" / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    source_motion["body_names"] = body_names
    (destination / "meta_motion.json").write_text(json.dumps(source_motion))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    args = parser.parse_args()
    reference = json.loads((args.reference_dataset / "meta_motion.json").read_text())
    body_names = reference["body_names"]
    for split in ("train", "val"):
        input_split = args.input_root / split
        if not input_split.exists():
            continue
        for source in sorted(path for path in input_split.iterdir() if path.is_dir()):
            destination = args.output_root / split / source.name
            if destination.exists():
                raise FileExistsError(destination)
            print(f"Projecting {source} -> {destination} ({len(body_names)} bodies)", flush=True)
            project_dataset(source, destination, body_names, args.chunk_rows)
    manifest_path = args.input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["name"] = manifest.get("name", args.input_root.name) + "-legacy27-body-schema"
    manifest["source_manifest"] = str(manifest_path.resolve())
    manifest["body_names"] = body_names
    manifest["schema_projection"] = (
        "Body position/orientation tensors projected by name onto the legacy "
        "root3 dataset's 27-body schema; all 29 joint channels are preserved."
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
