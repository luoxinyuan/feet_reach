"""Run frozen native SONIC on every fixed compliance tracking protocol clip."""
import os
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/compliance_eval/protocol_v1.json")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=160)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning  # noqa: F401
from isaaclab.app import AppLauncher

started = time.time()
manifest = json.loads(args.manifest.read_text())
rows = manifest["clips"]
if args.batch_size < 1:
    raise ValueError("batch-size must be positive")
if args.batch_size < len(rows):
    raise ValueError("Isaac Sim uses one simulation context per process; set --batch-size to at least the full clip count")
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
base_cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
OmegaConf.set_struct(base_cfg, False)
OmegaConf.set_readonly(base_cfg, False)
del state

app = AppLauncher(
    headless=True,
    device="cuda:0",
    enable_cameras=False,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=12 --/omni.tbb.globalcontrol/maxThreadCount=12",
).app

try:
    from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul
    from scripts.utils.helpers import make_env_policy

    # Keep the same reset and simulator conventions as the fixed compliance
    # tracking suite. This is pure reference tracking: no physical forces and
    # the learned force-compliance residual is held at zero.
    per_clip = [None] * len(rows)
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start:batch_start + args.batch_size]
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.checkpoint_path = str(args.checkpoint.resolve())
        cfg.vecnorm = "eval"
        cfg.seed = int(manifest["seeds"][0])
        cfg.task.num_envs = len(batch_rows)
        cfg.task.max_episode_length = 2000
        cfg.task.reward = {}
        cfg.task.termination = {}
        cfg.task.randomization = {}
        cfg.task.observe_on_reset = True
        cfg.task.viewer.resolution = [960, 540]
        cfg.eval_render = False
        cfg.headless = True
        paths = list(dict.fromkeys(row["dataset"] for row in batch_rows))
        cfg.task.command.dataset.mem_paths = paths
        cfg.task.command.dataset.path_weights = [1.0] * len(paths)
        cfg.task.command.dataset.resample_on_reset = False
        cfg.task.command.dataset.sample_once = True
        cfg.task.command.dataset.fix_motion_id = None
        cfg.task.command.dataset.fix_ds = None
        if hasattr(cfg.task.command.dataset, "motion_allowlists"):
            cfg.task.command.dataset.motion_allowlists = [None] * len(paths)
        cfg.task.command.guidance_enabled = False

        torch.manual_seed(cfg.seed + batch_start)
        np.random.seed(cfg.seed + batch_start)
        env, agent, _, _ = make_env_policy(cfg)
        env.eval()
        env.base_env.eval()
        agent.eval()
        cmd = env.command_manager
        robot = cmd.asset
        manager = env.action_manager
        if not hasattr(manager, "sonic") or not hasattr(manager, "history"):
            raise RuntimeError("Checkpoint did not construct native SONIC tracking adapter")
        if len(manager.joint_ids) != 29:
            raise RuntimeError(f"Expected 29 SONIC joints, got {len(manager.joint_ids)}")

        cmd.zero_init_prob = 1.0
        for key in cmd.init_noise_params:
            cmd.init_noise_params[key] = 0.0
        ds = cmd.dataset
        for i, row in enumerate(batch_rows):
            source = ds.datasets[paths.index(row["dataset"])]
            clip_index = int(row["clip"])
            st = int(source.starts[clip_index])
            end = st + int(row["frames"])
            idx = torch.arange(st, st + ds.max_step_size, device=source.starts.device).clamp_max(end - 1)
            ds._buf_A[i] = ds._to_float(source.data[idx].to(env.device), dtype=ds.buffer_float_dtype)
            ds._len_A[i] = int(row["frames"])
            for extra in ds.dataset_extra_keys:
                ds._info_A[extra["name"]][i] = source.info[extra["name"]][clip_index].to(env.device)

        td = env.reset()
        for i, row in enumerate(batch_rows):
            source = ds.datasets[paths.index(row["dataset"])]
            clip_index = int(row["clip"])
            st = int(source.starts[clip_index])
            expected_motion = int(source.data.motion_id[st].item())
            expected_step = int(source.data.step[st].item())
            if int(cmd._motion.motion_id[i, 0]) != expected_motion or int(cmd._motion.step[i, 0]) != expected_step:
                raise RuntimeError(f"Protocol clip mismatch at row {batch_start+i}: expected {(expected_motion, expected_step)}")

        sonic_joint_ids = torch.tensor(
            [cmd.dataset.joint_names.index(name) for name in manager.joint_names],
            device=env.device,
            dtype=torch.long,
        )
        original_forward = manager.sonic.forward
        reference_q = reference_qd = reference_orientation = None

        def forward_tracking(_q_reference, _orientation_6d, proprioception):
            return original_forward(reference_q, reference_orientation, proprioception, qd_reference=reference_qd)

        manager.sonic.forward = forward_tracking
        hand_names = ["left_hand_mimic", "right_hand_mimic"]
        hand_asset_ids = torch.tensor([robot.body_names.index(name) for name in hand_names], device=env.device)
        hand_motion_ids = torch.tensor([ds.body_names.index(name) for name in hand_names], device=env.device)
        lengths = cmd.lengths.long()
        max_length = int(lengths.max().item())
        sums = {"joint_mae_rad": torch.zeros(len(batch_rows), device=env.device),
                "root_xy_error_m": torch.zeros(len(batch_rows), device=env.device),
                "root_z_error_m": torch.zeros(len(batch_rows), device=env.device),
                "hand_position_error_m": torch.zeros(len(batch_rows), device=env.device)}
        counts = torch.zeros(len(batch_rows), device=env.device)
        fall_step = torch.full((len(batch_rows),), -1, dtype=torch.long, device=env.device)

        def accumulate(valid):
            root = robot.data.root_pos_w - env.scene.env_origins
            target_root = cmd._motion.root_pos_w[:, 0]
            q = robot.data.joint_pos[:, manager.joint_ids]
            target_q = cmd._motion.joint_pos[:, 0, sonic_joint_ids]
            hands = robot.data.body_pos_w[:, hand_asset_ids] - env.scene.env_origins[:, None]
            target_hands = cmd._motion.body_pos_w[:, 0, hand_motion_ids]
            sums["joint_mae_rad"].add_(((q-target_q).abs().mean(-1))*valid)
            sums["root_xy_error_m"].add_(((root[:, :2]-target_root[:, :2]).norm(dim=-1))*valid)
            sums["root_z_error_m"].add_(((root[:, 2]-target_root[:, 2]).abs())*valid)
            sums["hand_position_error_m"].add_(((hands-target_hands).norm(dim=-1).mean(-1))*valid)
            counts.add_(valid)
            tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            fallen = (root[:, 2] < 0.30) | (tilt > np.deg2rad(75)) | (~torch.isfinite(q).all(-1))
            new_fall = (fall_step < 0) & fallen & valid.bool()
            fall_step[new_fall] = cmd.t[new_fall].long()

        with torch.no_grad():
            accumulate(torch.ones(len(batch_rows), device=env.device))
            for frame in range(1, max_length):
                # The adapter acts before the command advances, so the target
                # window starts at the next reference frame, matching tracking72.
                start = (cmd.t.long() + 1).clamp_max(lengths - 1)
                future = ds.get_slice(torch.arange(len(batch_rows), device=env.device), start,
                                      steps=torch.arange(10, device=env.device))
                reference_q = future.joint_pos[:, :, sonic_joint_ids]
                reference_qd = future.joint_vel[:, :, sonic_joint_ids]
                relative_orientation = quat_mul(
                    quat_conjugate(robot.data.root_quat_w[:, None].expand(-1, 10, -1)), future.root_quat_w
                )
                reference_orientation = matrix_from_quat(relative_orientation)[..., :, :2].reshape(len(batch_rows), 10, 6)
                td["action"] = torch.zeros(len(batch_rows), manager.action_dim, device=env.device)
                td = env.step(td)["next"]
                valid = (frame < lengths).float()
                accumulate(valid)

        for i, row in enumerate(batch_rows):
            denom = max(float(counts[i]), 1.0)
            per_clip[batch_start+i] = {
                "split": row["split"], "dataset": row["dataset"], "clip": int(row["clip"]),
                "source_motion_id": int(row["source_motion_id"]), "source_start_step": int(row["source_start_step"]),
                "category": row["category"], "frames": int(counts[i].item()),
                "joint_mae_rad": float((sums["joint_mae_rad"][i] / denom).item()),
                "root_xy_error_m": float((sums["root_xy_error_m"][i] / denom).item()),
                "root_z_error_m": float((sums["root_z_error_m"][i] / denom).item()),
                "hand_position_error_m": float((sums["hand_position_error_m"][i] / denom).item()),
                "fall": bool(fall_step[i] >= 0),
                "first_fall_frame": int(fall_step[i].item()) if fall_step[i] >= 0 else None,
            }
        print(json.dumps({"completed": min(batch_start+len(batch_rows), len(rows)), "total": len(rows),
                          "elapsed_s": round(time.time()-started, 1)}), flush=True)
        env.close()

    def summarize(group):
        metrics = ["joint_mae_rad", "root_xy_error_m", "root_z_error_m", "hand_position_error_m"]
        frames = sum(x["frames"] for x in group)
        # Clip-weighted aggregate prevents long source motions from dominating;
        # report the corresponding pooled-frame means as an audit value.
        return {
            "clips": len(group), "frames": frames,
            "fall_clips": sum(x["fall"] for x in group),
            "clip_mean": {k: float(np.mean([x[k] for x in group])) for k in metrics},
            "frame_weighted_mean": {k: float(np.average([x[k] for x in group], weights=[x["frames"] for x in group])) for k in metrics},
        }

    result = {
        "protocol": manifest["version"], "checkpoint": str(args.checkpoint.resolve()),
        "controller": "pretrained frozen SONIC native tracking backend", "residual_student_used": False,
        "force_guidance": False, "reference": "10-frame future joint position/velocity and relative root orientation",
        "dt_s": float(manifest["dt"]), "batch_size": args.batch_size,
        "all_160": summarize(per_clip),
        "validation_128": summarize([x for x in per_clip if x["split"] == "validation"]),
        "train_diagnostic_32": summarize([x for x in per_clip if x["split"] == "train_diagnostic"]),
        "per_clip": per_clip, "wall_seconds": time.time()-started,
    }
    report_path = args.output / "sonic_tracking_suite.json"
    report_path.write_text(json.dumps(result, indent=2))
    print("SONIC_TRACKING_SUITE_COMPLETE", json.dumps({k: result[k] for k in ("all_160", "validation_128", "train_diagnostic_32")}), flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
