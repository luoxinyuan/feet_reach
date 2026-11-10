#!/usr/bin/env python3
"""Zero-residual tracking preflight for the 3KP Jacobian + SuperMimic stack."""
import argparse
import faulthandler
import json
import math
import os
import sys
import time
import signal
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--clips", type=int, default=5)
parser.add_argument("--category", choices=("crouch", "left_arm", "right_arm", "locomotion", "yaw"), default=None)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
faulthandler.register(signal.SIGUSR1, all_threads=True)

import numpy as np
import torch
from omegaconf import OmegaConf
from scripts.compliance_eval.eval_supermimic_isaaclab import prepare_manifest_subset

manifest_path = ROOT / "cfg/compliance_eval_v1_1.json"
all_rows = json.loads(manifest_path.read_text())["clips"]
# Cover isolated arms, locomotion, and crouching with a short fixed set.
chosen = []
categories = (args.category,) if args.category else ("crouch", "left_arm", "right_arm", "locomotion", "yaw")
for category in categories:
    chosen.append(next(row.copy() for row in all_rows if row["category"] == category))
chosen = chosen[:args.clips]
subset_root = prepare_manifest_subset(chosen, args.output / "motion_subset")
os.environ["MEMPATH"] = str(subset_root)
print(f"prepared {len(chosen)} fixed clips in {subset_root}", flush=True)

from hydra import compose, initialize_config_dir
with initialize_config_dir(version_base=None, config_dir=str(ROOT / "cfg")):
    cfg = compose(config_name="train", overrides=[
        "task=G1/G1_supermimic_online_estimated_root3_crouch20_force_history",
        "algo=ceer_dense_force",
    ])
OmegaConf.set_struct(cfg, False)
OmegaConf.set_readonly(cfg, False)
OmegaConf.resolve(cfg)
n = len(chosen)
cfg.task.num_envs = n
cfg.task.max_episode_length = 2000
cfg.task.guidance_enabled = False
cfg.task.command.guidance_enabled = False
cfg.task.reward = {}
cfg.task.termination = {}
cfg.task.randomization = {}
cfg.task.sim.step_dt = 0.02
cfg.task.sim.isaac_physics_dt = 0.005
paths = list(dict.fromkeys(row["_subset_dataset"] for row in chosen))
cfg.task.command.dataset.mem_paths = paths
cfg.task.command.dataset.path_weights = [1.0] * len(paths)
cfg.task.command.dataset.resample_on_reset = False
cfg.task.command.dataset.sample_once = True
cfg.task.command.dataset.fix_motion_id = None
cfg.task.command.dataset.fix_ds = None
cfg.seed = 20261109
cfg.headless = True
cfg.app.headless = True

from isaaclab.app import AppLauncher
device = torch.device(args.device)
app = AppLauncher(headless=True, device=str(device), enable_cameras=False).app
try:
    print("constructing SimpleEnv", flush=True)
    from active_adaptation.envs import SimpleEnv
    env = SimpleEnv(cfg.task)
    print("SimpleEnv constructed", flush=True)
    env.eval()
    cmd, robot, manager = env.command_manager, env.robot, env.action_manager
    if manager.action_dim != 29 or len(manager.joint_names) != 29:
        raise RuntimeError(f"Expected 29 SuperMimic joints, got {manager.action_dim}")
    if len(manager.low_policy.encoder) != 5 or manager.low_policy.goal_scale.numel() != 680:
        raise RuntimeError("Released SuperMimic actor dimensions do not match deploy interface")
    # Pin each environment to one protocol clip, using the same fixed-frame store
    # construction as the full SuperMimic benchmark.
    for i, row in enumerate(chosen):
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
    print("resetting fixed motions", flush=True)
    td = env.reset()
    print("environment reset complete", flush=True)
    counts = torch.zeros(n, device=device)
    metrics = {k: torch.zeros(n, device=device) for k in ("joint_mae", "root_xy", "hand")}
    fallen = torch.zeros(n, dtype=torch.bool, device=device)
    first_fall = torch.full((n,), -1, dtype=torch.long, device=device)
    qids = [cmd.dataset.joint_names.index(name) for name in manager.policy_joint_names]
    hand_ids = [cmd.dataset.body_names.index(f"{side}_hand_mimic") for side in ("left", "right")]
    asset_hand_ids = [robot.body_names.index(f"{side}_hand_mimic") for side in ("left", "right")]
    frame_limit = max(int(row["frames"]) for row in chosen)

    def record(frame):
        ref = cmd.dataset.get_slice(None, cmd.t, steps=1)
        q = robot.data.joint_pos[:, manager.joint_ids]
        rq = ref.joint_pos[:, 0, qids]
        z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        ref_z = ref.root_pos_w[:, 0, 2]
        quat = robot.data.root_quat_w
        up_z = 1. - 2. * (quat[:, 1].square() + quat[:, 2].square())
        # Crouch references can intentionally put the pelvis below 0.30 m.
        # Treat a low absolute height as a fall only if the robot has dropped
        # well below its own reference; the tilt criterion remains absolute.
        fail = (z < .10) | (z < ref_z - .15) | (up_z < math.cos(math.radians(75))) | ~torch.isfinite(q).all(-1)
        active = frame < torch.tensor([r["frames"] for r in chosen], device=device)
        new = active & fail & ~fallen
        first_fall[new] = frame
        fallen.logical_or_(new)
        active &= ~fallen
        if not active.any():
            return
        root_ref = ref.root_pos_w[:, 0] + env.scene.env_origins
        hand_ref = ref.body_pos_w[:, 0, hand_ids] + env.scene.env_origins[:, None]
        metrics["joint_mae"][active] += (q - rq).abs().mean(-1)[active]
        metrics["root_xy"][active] += (robot.data.root_pos_w[:, :2] - root_ref[:, :2]).norm(dim=-1)[active]
        metrics["hand"][active] += (robot.data.body_pos_w[:, asset_hand_ids] - hand_ref).norm(dim=-1).mean(-1)[active]
        counts[active] += 1

    record(0)
    started = time.time()
    print(f"beginning {frame_limit - 1} tracking frames", flush=True)
    with torch.inference_mode():
        for frame in range(frame_limit - 1):
            td["action"] = torch.zeros(n, 29, device=device)
            td = env.step(td)["next"]
            record(frame + 1)
            if (frame + 1) % 100 == 0:
                print(f"preflight frames={frame+1}/{frame_limit-1}", flush=True)
    per_clip = []
    for i, row in enumerate(chosen):
        d = float(counts[i])
        per_clip.append({
            "category": row["category"], "dataset": row["dataset"], "clip": row["clip"],
            "frames": row["frames"], "scored_frames": int(d), "fall": bool(fallen[i]),
            "first_fall_frame": None if int(first_fall[i]) < 0 else int(first_fall[i]),
            "joint_mae_rad": None if d == 0 else float(metrics["joint_mae"][i] / d),
            "root_xy_error_m": None if d == 0 else float(metrics["root_xy"][i] / d),
            "hand_error_m": None if d == 0 else float(metrics["hand"][i] / d),
        })
    scored = [x for x in per_clip if x["scored_frames"]]
    summary = {
        "clips": len(per_clip), "fall_clips": sum(x["fall"] for x in per_clip),
        "clip_mean_joint_mae_rad": float(np.mean([x["joint_mae_rad"] for x in scored])),
        "clip_mean_root_xy_error_m": float(np.mean([x["root_xy_error_m"] for x in scored])),
        "clip_mean_hand_error_m": float(np.mean([x["hand_error_m"] for x in scored])),
    }
    result = {
        "controller": "released SuperMimic checkpoint, zero PPO residual, 3KP online Jacobian command disabled",
        "checkpoint": "/home/xl521/supermimic_force_transition/pretrained/e11c_0220000/ckpt_e11c_0220000.pt",
        "dt_s": env.step_dt, "interface": {"motion_goal": "10 x 68 = 680", "proprio_history": "term-major 10 x 93 = 930", "joints": 29},
        "summary": summary, "per_clip": per_clip, "wall_seconds": time.time() - started,
    }
    (args.output / "preflight.json").write_text(json.dumps(result, indent=2))
    print("SUPERMIMIC_JACOBIAN_PREFLIGHT " + json.dumps(summary), flush=True)
    env.close()
except BaseException:
    error = traceback.format_exc()
    print(error, flush=True)
    (args.output / "preflight_error.txt").write_text(error)
    raise
finally:
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
    except Exception:
        pass
    app.close()
