#!/usr/bin/env python3
"""Evaluate released SuperMimic in the feet_reach IsaacLab G1 environment.

Uses the fixed compliance tracking manifest and G1 asset/PD gains. The released
policy runs at 50 Hz; its normalized joint actions are mapped to the native
joint-position action manager. The simulator and reset path are the same ones
used by the regular compliance tracking evaluator.
"""
import argparse
import faulthandler
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SM_ROOT = Path("/home/xl521/supermimic_force_transition")
sys.path.insert(0, str(SM_ROOT / "code"))

from torch import nn
from supermimic.rotations import heading_mat, mtv, quat_to_mat


class ReleasedPolicy(nn.Module):
    def __init__(self, state):
        super().__init__()
        p = state["policy"]
        self.encoder = nn.ModuleList([nn.Linear(680, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 32)])
        self.decoder = nn.ModuleList([nn.Linear(962, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 29)])
        for i, layer in enumerate(self.encoder):
            layer.weight.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.bias"])
        for i, layer in enumerate(self.decoder):
            layer.weight.data.copy_(p[f"decoder.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"decoder.mlp.net.{2*i}.bias"])
        self.register_buffer("goal_scale", p["encoders.motion.scale"].clone())
        self.register_buffer("obs_scale", p["decoder.scale"].clone())
        self.eval()

    def forward(self, obs, goal):
        x = goal * self.goal_scale
        for layer in self.encoder[:-1]:
            x = torch.nn.functional.silu(layer(x))
        z = torch.tanh(self.encoder[-1](x))
        x = torch.cat((z, obs * self.obs_scale), dim=-1)
        for layer in self.decoder[:-1]:
            x = torch.nn.functional.silu(layer(x))
        return self.decoder[-1](x)


def flatten_policy_history(obs_hist):
    chunks = []
    start = 0
    for width in (29, 29, 3, 3, 29):
        chunks.append(obs_hist[..., start:start + width].flatten(1))
        start += width
    return torch.cat(chunks, dim=-1)


def build_goal(cmd, robot, env, deploy, q0, qids, starts):
    steps = torch.arange(1, 11, device=robot.device)
    ref = cmd.dataset.get_slice(None, starts, steps=steps)
    pos = ref.root_pos_w + env.scene.env_origins[:, None, :]
    rot = quat_to_mat(ref.root_quat_w)
    state_pos = robot.data.root_pos_w
    state_rot = quat_to_mat(robot.data.root_quat_w)
    dp = mtv(state_rot[:, None], pos - state_pos[:, None])
    gravity_w = torch.zeros_like(ref.root_lin_vel_w)
    gravity_w[..., 2] = -1.0
    gravity_body = mtv(rot, gravity_w)
    ang_body = mtv(rot, ref.root_ang_vel_w)
    lin_heading = mtv(heading_mat(rot), ref.root_lin_vel_w)
    q = ref.joint_pos[..., qids]
    qd = ref.joint_vel[..., qids]
    goal = torch.cat([q - q0, qd, gravity_body, ang_body, lin_heading, pos[..., 2:3]], dim=-1)
    if goal.shape[-1] != 68:
        raise RuntimeError(f"Expected 68 goal features per frame, got {goal.shape[-1]}")
    return goal.flatten(1)


def current_obs(robot, deploy_names, asset_to_policy, q0, previous_action):
    q = robot.data.joint_pos[:, asset_to_policy]
    qd = robot.data.joint_vel[:, asset_to_policy]
    gyro = robot.data.root_ang_vel_b
    gravity = robot.data.projected_gravity_b if hasattr(robot.data, "projected_gravity_b") else None
    if gravity is None:
        from isaaclab.utils.math import quat_apply_inverse
        gravity_w = torch.zeros_like(robot.data.root_ang_vel_w)
        gravity_w[:, 2] = -1.0
        gravity = quat_apply_inverse(robot.data.root_quat_w, gravity_w)
    return torch.cat((q - q0, qd, gyro, gravity, previous_action), dim=-1)


def prepare_manifest_subset(rows, output_root):
    """Build small TensorDict motion stores containing only fixed eval clips."""
    import shutil

    source_root = Path(os.environ.get("MEMPATH", ROOT / "dataset")).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    torch_dtype_to_numpy = {"torch.float16": np.float16, "torch.float32": np.float32,
                            "torch.int32": np.int32, "torch.int64": np.int64}
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset_index, (dataset_name, dataset_rows) in enumerate(by_dataset.items()):
        subset_name = f"eval_dataset_{dataset_index:03d}"
        for row in dataset_rows:
            row["_subset_dataset"] = subset_name
        src = (source_root / dataset_name).resolve()
        motion_meta = json.loads((src / "meta_motion.json").read_text())
        td_meta = json.loads((src / "_tensordict" / "meta.json").read_text())
        entries = []
        total = 0
        for new_clip, row in enumerate(dataset_rows):
            old_clip = int(row["clip"])
            start = int(motion_meta["starts"][old_clip])
            length = int(row["frames"])
            end = int(motion_meta["ends"][old_clip])
            if start + length > end:
                raise ValueError(f"Manifest clip exceeds source interval: {row}")
            row["_subset_clip"] = new_clip
            entries.append((start, length, total))
            total += length
        dest = output_root / subset_name / "_tensordict"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "meta.json", output_root / subset_name / "meta.json")
        new_td_meta = json.loads(json.dumps(td_meta))
        for key, spec in td_meta.items():
            if key in ("shape", "device", "_type"):
                continue
            shape = tuple(spec["shape"])
            dtype = torch_dtype_to_numpy[spec["dtype"]]
            source = np.memmap(src / "_tensordict" / f"{key}.memmap", mode="r", dtype=dtype, shape=shape)
            target_shape = (total, *shape[1:])
            target = np.memmap(dest / f"{key}.memmap", mode="w+", dtype=dtype, shape=target_shape)
            for start, length, target_start in entries:
                target[target_start:target_start + length] = source[start:start + length]
            target.flush()
            new_td_meta[key]["shape"] = list(target_shape)
            del source, target
        new_td_meta["shape"] = [total]
        (dest / "meta.json").write_text(json.dumps(new_td_meta))
        new_meta = {"body_names": motion_meta["body_names"], "joint_names": motion_meta["joint_names"],
                    "starts": [], "ends": [], "info": {}}
        for start, length, target_start in entries:
            new_meta["starts"].append(target_start)
            new_meta["ends"].append(target_start + length)
        (output_root / subset_name / "meta_motion.json").write_text(json.dumps(new_meta))
    return output_root


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "cfg/compliance_eval_v1_1.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Optional leading clip subset for an adapter smoke check")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(a.manifest.read_text())["clips"]
    if a.limit:
        rows = rows[:a.limit]
    n = len(rows)
    subset_root = prepare_manifest_subset(rows, a.output / "motion_subset")
    os.environ["MEMPATH"] = str(subset_root)
    deploy = json.loads((a.checkpoint.parent / "deploy_config.json").read_text())
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(a.device)
    policy = ReleasedPolicy(ck).to(device).eval()
    q0 = torch.tensor(deploy["default_joint_pos"], dtype=torch.float32, device=device)
    policy_names = list(deploy["joint_names"])
    scales_by_name = dict(zip(policy_names, deploy["action_scale"]))

    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "cfg")):
        cfg = compose(config_name="train", overrides=["task=G1/G1_base", "algo=ceer_dense_force"])
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    cfg.task.num_envs = n
    cfg.task.max_episode_length = 2000
    cfg.task.reward = {}
    cfg.task.termination = {}
    cfg.task.randomization = {}
    cfg.task.sim.step_dt = 0.02
    cfg.task.sim.isaac_physics_dt = 0.005
    paths = list(dict.fromkeys(r["_subset_dataset"] for r in rows))
    cfg.task.command.dataset.mem_paths = paths
    cfg.task.command.dataset.path_weights = [1.0] * len(paths)
    cfg.task.command.dataset.resample_on_reset = False
    cfg.task.command.dataset.sample_once = True
    cfg.task.command.dataset.fix_motion_id = None
    cfg.task.command.dataset.fix_ds = None
    cfg.task.command.init_noise = {k: 0.0 for k in cfg.task.command.init_noise.keys()}
    cfg.task.action.action_scaling = scales_by_name
    cfg.task.action.max_delay = 0
    cfg.task.action.alpha = [1.0, 1.0]
    cfg.task.action.alpha_wide = [1.0, 1.0]
    cfg.task.action.alpha_jit_scale = None
    cfg.task.action.boot_protect = False
    cfg.seed = a.seed
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=str(device), enable_cameras=False).app
    try:
        print("[adapter] importing feet_reach IsaacLab env", flush=True)
        from active_adaptation.envs import SimpleEnv
        print("[adapter] importing done", flush=True)
        print(f"[adapter] creating IsaacLab env: num_envs={n}, MEMPATH={os.environ['MEMPATH']}", flush=True)
        try:
            env = SimpleEnv(cfg.task)
        except BaseException:
            traceback.print_exc()
            os._exit(2)
        print("[adapter] IsaacLab env constructed", flush=True)
        robot = env.robot
        cmd = env.command_manager
        manager = env.action_manager
        asset_names = list(robot.joint_names)
        if set(asset_names) != set(policy_names) or len(asset_names) != len(policy_names):
            raise RuntimeError(f"G1 asset joints differ from SuperMimic deploy joints: asset={asset_names}, policy={policy_names}")
        policy_asset_ids = [asset_names.index(name) for name in policy_names]
        if manager.action_dim != len(policy_names):
            raise RuntimeError(f"Expected 29 position targets, got action_dim={manager.action_dim}")
        # The feet_reach asset's standing default differs from SuperMimic's
        # action origin. Compensate in the native joint-position action offset
        # so its target remains q0 + action_scale * action.
        action_policy_ids = [policy_names.index(name) for name in manager.joint_names]
        manager.offset[:] = 0.0
        manager.offset[:, manager.joint_ids] = q0[action_policy_ids] - manager.default_joint_pos[:, manager.joint_ids]
        qdiff = float((manager.default_joint_pos[:, manager.joint_ids] + manager.offset[:, manager.joint_ids] - q0[action_policy_ids]).abs().max().item())
        if qdiff > 1e-4:
            raise RuntimeError(f"IsaacLab action target origin differs from SuperMimic deploy config: {qdiff}")
        expected_asset_scale = torch.tensor([scales_by_name[name] for name in manager.joint_names], device=device)
        scale_diff = float((manager.action_scaling - expected_asset_scale).abs().max().item())
        if scale_diff > 1e-4:
            raise RuntimeError(f"IsaacLab action scaling differs from deploy config: {scale_diff}")

        for i, row in enumerate(rows):
            ds = cmd.dataset.datasets[paths.index(row["_subset_dataset"])]
            clip_i = int(row["_subset_clip"])
            start = int(ds.starts[clip_i])
            end = start + int(row["frames"])
            idx = torch.arange(start, start + cmd.dataset.max_step_size, device=ds.starts.device).clamp_max(end - 1)
            cmd.dataset._buf_A[i] = cmd.dataset._to_float(ds.data[idx].to(device), dtype=cmd.dataset.buffer_float_dtype)
            cmd.dataset._len_A[i] = int(row["frames"])
        cmd.zero_init_prob = 1.0
        for key in cmd.init_noise_params:
            cmd.init_noise_params[key] = 0.0

        td = env.reset()
        env.episode_length_buf.zero_()
        initial_ref = cmd.dataset.get_slice(None, cmd.t, steps=1)
        for i, row in enumerate(rows):
            if int(initial_ref.motion_id[i, 0]) != int(row["source_motion_id"]):
                raise RuntimeError(f"Manifest motion ID mismatch at env {i}: got={int(initial_ref.motion_id[i, 0])}, expected={row['source_motion_id']}, dataset={row['dataset']}, clip={row['clip']}, subset={row['_subset_dataset']}:{row['_subset_clip']}")
            if int(initial_ref.step[i, 0]) != int(row["source_start_step"]):
                raise RuntimeError(f"Manifest start frame mismatch at env {i}: got={int(initial_ref.step[i, 0])}, expected={row['source_start_step']}")
        qids = [cmd.dataset.joint_names.index(name) for name in policy_names]
        hand_ids = [cmd.dataset.body_names.index(f"{side}_hand_mimic") for side in ("left", "right")]
        prev_action = torch.zeros(n, len(policy_names), device=device)
        obs = current_obs(robot, policy_names, policy_asset_ids, q0, prev_action)
        history = obs[:, None, :].expand(-1, 10, -1).contiguous()
        metrics = {k: torch.zeros(n, device=device) for k in ("joint_mae", "root_xy", "root_z", "hand")}
        counts = torch.zeros(n, device=device)
        fallen = torch.zeros(n, dtype=torch.bool, device=device)
        first_fall = torch.full((n,), -1, dtype=torch.long, device=device)

        def record(frame_no):
            ref = cmd.dataset.get_slice(None, cmd.t, steps=1)
            ref_q = ref.joint_pos[:, 0, qids]
            active = (frame_no < torch.as_tensor([r["frames"] for r in rows], device=device)) & ~fallen
            q = robot.data.joint_pos[:, policy_asset_ids]
            quat = robot.data.root_quat_w
            up_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
            fail = (robot.data.root_pos_w[:, 2] < 0.30) | (up_z < math.cos(math.radians(75.0))) | ~torch.isfinite(q).all(-1)
            newly = fail & active
            first_fall[newly] = frame_no[newly]
            fallen.logical_or_(newly)
            active &= ~newly
            if not active.any():
                return
            ref_root = ref.root_pos_w[:, 0] + env.scene.env_origins
            ref_hand = ref.body_pos_w[:, 0, hand_ids] + env.scene.env_origins[:, None, :]
            hand_asset_ids = [robot.body_names.index(f"{side}_hand_mimic") for side in ("left", "right")]
            dq = (q[:, :] - ref_q).abs().mean(-1)
            root_err = robot.data.root_pos_w - ref_root
            hand_err = (robot.data.body_pos_w[:, hand_asset_ids] - ref_hand).norm(dim=-1).mean(-1)
            for key, val in (("joint_mae", dq), ("root_xy", root_err[:, :2].norm(dim=-1)), ("root_z", root_err[:, 2].abs()), ("hand", hand_err)):
                metrics[key][active] += val[active]
            counts[active] += 1

        record(torch.zeros(n, dtype=torch.long, device=device))
        max_frames = max(int(r["frames"]) for r in rows)
        t0 = time.time()
        with torch.inference_mode():
            for frame_no in range(max_frames - 1):
                goal = build_goal(cmd, robot, env, deploy, q0, qids, cmd.t + 1)
                action = policy(flatten_policy_history(history), goal)
                prev_action = action
                asset_action_ids = [policy_names.index(name) for name in manager.joint_names]
                td["action"] = action[:, asset_action_ids]
                td = env.step(td)["next"]
                obs = current_obs(robot, policy_names, policy_asset_ids, q0, prev_action)
                history = torch.cat((history[:, 1:], obs[:, None, :]), dim=1)
                record(torch.full((n,), frame_no + 1, dtype=torch.long, device=device))
                if (frame_no + 1) % 100 == 0:
                    print(f"frames={frame_no + 1}/{max_frames - 1} elapsed_s={time.time() - t0:.1f}", flush=True)

        per_clip = []
        for i, row in enumerate(rows):
            denom = float(counts[i].item())
            per_clip.append({
                "split": row["split"], "dataset": row["dataset"], "clip": row["clip"],
                "category": row["category"], "frames": int(row["frames"]),
                "seed": a.seed, "scored_frames": int(denom), "fall": bool(fallen[i].item()),
                "first_fall_frame": None if int(first_fall[i]) < 0 else int(first_fall[i]),
                "full_joint_mae_rad": None if denom == 0 else float(metrics["joint_mae"][i].item() / denom),
                "root_xy_error_m": None if denom == 0 else float(metrics["root_xy"][i].item() / denom),
                "root_z_error_m": None if denom == 0 else float(metrics["root_z"][i].item() / denom),
                "hand_position_error_m": None if denom == 0 else float(metrics["hand"][i].item() / denom),
            })
        def summary(records):
            if not records:
                return {"clips": 0, "frames": 0, "fall_clips": 0}
            keys = ("full_joint_mae_rad", "root_xy_error_m", "root_z_error_m", "hand_position_error_m")
            scored = [x for x in records if x["scored_frames"] > 0]
            frame_count = sum(x["scored_frames"] for x in scored)
            return {"clips": len(records), "scored_clips": len(scored), "frames": frame_count,
                    "fall_clips": sum(x["fall"] for x in records),
                    "clip_mean": {k: (None if not scored else float(np.mean([x[k] for x in scored]))) for k in keys},
                    "frame_weighted_mean": {k: (None if not frame_count else float(sum(x[k] * x["scored_frames"] for x in scored) / frame_count)) for k in keys}}
        val = [x for x in per_clip if x["split"] == "validation"]
        result = {"checkpoint": str(a.checkpoint), "seed": a.seed, "simulator_backend": "feet_reach IsaacLab / PhysX",
                  "protocol": str(a.manifest), "clip_count": n, "dt_s": 0.02, "physical_force": False,
                  "domain_randomization": False, "control": "G1 position targets, IsaacLab asset PD gains; SuperMimic deploy action scale",
                  "initial_root_lift_m": 0.04, "all_clips": summary(per_clip), "validation": summary(val),
                  "per_clip": per_clip, "wall_seconds": time.time() - t0}
        (a.output / "supermimic_isaaclab_tracking.json").write_text(json.dumps(result, indent=2))
        print("SUPERMIMIC_ISAACLAB_COMPLETE " + json.dumps({"all_clips": result["all_clips"], "validation": result["validation"], "wall_seconds": result["wall_seconds"]}), flush=True)
        env.close()
    finally:
        if sys.exc_info()[0] is not None:
            traceback.print_exc()
            os._exit(3)
        app.close()


if __name__ == "__main__":
    main()
