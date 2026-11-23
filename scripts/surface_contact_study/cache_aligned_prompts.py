"""Precompute calibrated eight-point aligned force replays for RL datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from active_adaptation.utils.segmented_guidance import rot
from active_adaptation.utils.surface_fk import SurfaceFK, quat_apply
from scripts.compliance_eval.protocol import load_arrays
from scripts.surface_contact_study.fixed8_learned_force import Refiner, readout, refine_parameters, select_windows
from scripts.surface_contact_study.masked_spring import load_study


PATHS = ["passive_whole_body_full/train", "crouch_lowlevel/train"]


def init(args):
    if args.study is not None:
        manifest, geometry, points, _ = load_study(args.study)
    else:
        manifest = json.loads((args.dataset / PATHS[0] / "meta_motion.json").read_text())
        geometry = json.loads(args.geometry.read_text())
        points = json.loads(args.points.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    digests = []
    for slot, path in enumerate(PATHS):
        source = args.dataset / path / "meta_motion.json"
        digests.append(hashlib.sha256(source.read_bytes()).hexdigest())
        meta = json.loads(source.read_text())
        length = np.minimum(1000, np.asarray(meta["ends"]) - meta["starts"])
        n = min(len(length), args.max_clips) if args.max_clips else len(length)
        offsets = np.full(len(length), -1, dtype=np.int64)
        offsets[:n] = np.r_[0, np.cumsum(length[:n])[:-1]]
        np.save(args.output / f"offsets{slot}.npy", offsets)
        np.lib.format.open_memmap(args.output / f"forces{slot}.npy", mode="w+", dtype=np.float32,
                                 shape=(int(length[:n].sum()), 8, 3)).flush()
    (args.output / "geometry.json").write_text(json.dumps(geometry))
    meta = dict(version="fixed8-aligned-observed-v1", paths=PATHS, source_sha256=digests,
                points=points, joint_names=manifest["joint_names"], workers=args.workers,
                max_clips=args.max_clips, solver=str(args.solver.resolve()),
                solver_sha256=hashlib.sha256(args.solver.read_bytes()).hexdigest())
    (args.output / "build.json").write_text(json.dumps(meta, indent=2))


class Planner:
    def __init__(self, saved, points, geometry, joint_names):
        self.saved = saved
        self.fk = SurfaceFK(geometry, joint_names, "cuda")
        self.body_ids = [self.fk.bodies.index(p["body"]) for p in points]
        self.local = self.cuda([p["offset"] for p in points])
        self.generator = Refiner(33 + 41 + 24, 0, 1.5, saved["hidden"]).cuda()
        self.generator.load_state_dict(saved["generator"])
        self.generator.eval().requires_grad_(False)
        self.reader = readout(33 + 24, saved["hidden"], saved["reader"], "cuda")
        self.weights = self.cuda(saved["group_weights"])

    @staticmethod
    def cuda(values):
        return torch.as_tensor(values, dtype=torch.float32, device="cuda")

    def __call__(self, q, pos, quat, length, order):
        n = len(q)
        starts = np.r_[0, np.cumsum(length)[:-1]]
        ends = starts + length
        indices = np.arange(n)
        previous = np.maximum(indices - 1, np.repeat(starts, length))
        last = np.repeat(ends - 1, length)
        saved = self.saved
        rotation = rot(quat).astype(np.float32)
        q_ordered = q[:, order]
        state = np.c_[q_ordered, -rotation[:, 2, :], pos[:, 2]]
        labels = [q_ordered[np.minimum(indices + 1, last)] - q_ordered]
        for h in (1, 5):
            following = np.minimum(indices + h, last)
            translation = ((pos[following] - pos)[:, None] @ rotation)[:, 0]
            relative = rotation.transpose(0, 2, 1) @ rotation[following]
            labels.extend([translation, Rotation.from_matrix(relative).as_rotvec()])
        target = np.concatenate(labels, axis=-1).astype(np.float32)
        energy = np.square(target / saved["activity_scale"]).sum(-1)
        blocks = [(t, min(t + saved["block"], end))
                  for start, end in zip(starts, ends) for t in range(start, end, saved["block"])]
        blocks = np.asarray([(a, b, int(1 + np.searchsorted(saved["thresholds"], energy[a:b].mean())))
                             for a, b in blocks], dtype=np.int64)
        pose = self.cuda((state - saved["state_mean"]) / saved["state_scale"])
        motion = self.cuda(target / saved["target_scale"])
        rq = self.cuda(quat)
        with torch.no_grad():
            location, orientation = self.fk(self.cuda(q), self.cuda(pos),
                                            torch.nn.functional.normalize(rq, dim=-1))
            location = location[:, self.body_ids] + quat_apply(
                orientation[:, self.body_ids], self.local.expand(n, -1, -1))
            future = np.minimum(previous + saved["horizon"], last)
            R = self.cuda(rotation)
            delta = (location[future] - location[previous]) @ R
            source = torch.cat((pose[previous], motion.clamp(-8., 8.),
                                (delta.flatten(1) * 40.).clamp(-4., 4.)), 1)
            envelope = torch.ones(n, device="cuda")
            envelope[starts] = 0.
            mask = torch.ones((n, 8), dtype=torch.bool, device="cuda")
            initial = self.generator.net(source)
            dense = self.generator.force_from_parameters(initial, delta, mask, envelope)
            mask = select_windows(torch.as_tensor(blocks, device="cuda"), mask, dense, pose,
                                  motion, self.weights, self.reader, 0, compact=True, reset=starts,
                                  length=saved["block"], probe_steps=saved["probe_steps"],
                                  retain_common=saved["retain_common"])
            prior = self.generator.force_from_parameters(initial, delta, mask, envelope)
        f, _ = refine_parameters(self.generator, self.reader, initial, delta, mask, envelope, pose,
                                 motion, self.weights, prior, saved["inner_steps"], batch_size=8192)
        world = f @ R.transpose(1, 2)
        assert torch.isfinite(world).all()
        assert (world.norm(dim=-1) > 1e-6).sum(-1).max() <= 3
        return world.cpu().numpy()


def worker(args, metadata):
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    saved = torch.load(metadata["solver"], map_location=device, weights_only=True)
    geometry = json.loads((args.output / "geometry.json").read_text())
    completed = 0
    start = time.monotonic()
    for slot, path in enumerate(metadata["paths"]):
        meta, arrays = load_arrays(args.dataset / path)
        offsets = np.load(args.output / f"offsets{slot}.npy")
        forces = np.lib.format.open_memmap(args.output / f"forces{slot}.npy", mode="r+")
        planner = Planner(saved, metadata["points"], geometry, meta["joint_names"])
        order = [meta["joint_names"].index(j) for j in metadata["joint_names"]]
        ids = np.flatnonzero(offsets >= 0)[args.worker::metadata["workers"]]
        length = np.minimum(1000, np.asarray(meta["ends"]) - meta["starts"])
        for group in np.split(ids, np.unique(np.searchsorted(np.cumsum(length[ids]),
                                np.arange(args.batch_frames, length[ids].sum(), args.batch_frames)))):
            if not len(group):
                continue
            slices = [slice(meta["starts"][i], meta["starts"][i] + length[i]) for i in group]
            values = [np.concatenate([arrays[key][s] for s in slices]).astype(np.float32)
                      for key in ("joint_pos", "root_pos_w", "root_quat_w")]
            batch = planner(*values, length[group], order)
            position = 0
            for i in group:
                size = length[i]
                forces[offsets[i]:offsets[i] + size] = batch[position:position + size]
                position += size
            completed += position
            forces.flush()
            print(f"rank {args.worker} {path}: {completed:,} frames in {time.monotonic()-start:.1f}s",
                  flush=True)
    (args.output / f"worker{args.worker}.done").write_text(str(completed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--batch-frames", type=int, default=49152)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.init:
        if args.solver is None or (args.study is None and (args.geometry is None or args.points is None)):
            parser.error("--init requires --solver and either --study or --geometry and --points")
        init(args)
        return
    meta = json.loads((args.output / "build.json").read_text())
    if args.finalize:
        total = sum(int((args.output / f"worker{i}.done").read_text()) for i in range(meta["workers"]))
        expected = sum(len(np.load(args.output / f"forces{i}.npy", mmap_mode="r"))
                       for i in range(len(meta["paths"])))
        if total != expected:
            raise RuntimeError(f"Force cache contains {total:,} of {expected:,} planned frames")
        p = args.output / "manifest.json.tmp"
        p.write_text(json.dumps(dict(meta, frames=total), indent=2))
        p.replace(args.output / "manifest.json")
        print(f"Ready: {total:,} frames")
    else:
        worker(args, meta)


if __name__ == "__main__":
    main()
