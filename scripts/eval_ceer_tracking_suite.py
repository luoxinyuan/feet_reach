"""Replay fixed whole-body tracking clips using a native CEER checkpoint."""
import os
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
import argparse, json, os as _os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--manifest", type=Path, default=ROOT / "artifacts/compliance_eval/protocol_v1.json")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--clip-row", type=int, help="Run only this manifest row and record a tracking video")
p.add_argument("--source-dataset", help="Run a dataset-local clip, e.g. passive_whole_body_full/val")
p.add_argument("--source-clip", type=int, help="Dataset-local clip index; requires --source-dataset")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

started = time.time()
manifest = json.loads(a.manifest.read_text())
all_rows = manifest["clips"]
if a.source_dataset is not None:
    if a.source_clip is None:
        p.error("--source-dataset requires --source-clip")
    meta = json.loads((ROOT / "dataset" / a.source_dataset / "meta_motion.json").read_text())
    ci = int(a.source_clip)
    s0, e0 = int(meta["starts"][ci]), int(meta["ends"][ci])
    rows = [{"split": "validation", "dataset": a.source_dataset, "clip": ci,
             "source_start_step": 0, "frames": min(e0-s0, 1000), "category": "tracking72"}]
elif a.clip_row is not None:
    rows = [all_rows[a.clip_row]]
else:
    rows = all_rows
state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
base_cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
OmegaConf.set_struct(base_cfg, False)
OmegaConf.set_readonly(base_cfg, False)
del state
paths = list(dict.fromkeys(row["dataset"] for row in rows))
base_cfg.checkpoint_path = str(a.checkpoint.resolve())
base_cfg.vecnorm = "eval"
base_cfg.seed = int(manifest["seeds"][0])
# Older CEER snapshots predate these eval-inert PPO config defaults.
for key, value in {"symmetry_augmentation": False, "train_estimators": True,
                   "student_use_gt_joint_target": False, "geometry_points": 0}.items():
    if key not in base_cfg.algo:
        base_cfg.algo[key] = value
base_cfg.task.num_envs = len(rows)
base_cfg.task.max_episode_length = 2000
base_cfg.task.reward = {}
base_cfg.task.termination = {}
base_cfg.task.randomization = {}
base_cfg.task.observe_on_reset = True
base_cfg.task.viewer.resolution = [960, 540]
make_video = a.clip_row is not None or a.source_dataset is not None
base_cfg.eval_render = make_video
base_cfg.headless = True
base_cfg.task.command.dataset.mem_paths = paths
base_cfg.task.command.dataset.path_weights = [1.0] * len(paths)
base_cfg.task.command.dataset.resample_on_reset = False
base_cfg.task.command.dataset.sample_once = True
base_cfg.task.command.dataset.fix_motion_id = int(a.source_clip) if a.source_dataset is not None else None
base_cfg.task.command.dataset.fix_ds = 0 if a.source_dataset is not None else None
if hasattr(base_cfg.task.command.dataset, "motion_allowlists"):
    base_cfg.task.command.dataset.motion_allowlists = [None] * len(paths)
if "guidance_enabled" in base_cfg.task.command:
    base_cfg.task.command.guidance_enabled = False

app = AppLauncher(headless=True, device="cuda:0", enable_cameras=make_video,
                  kit_args="--/plugins/carb.tasking.plugin/threadCount=12 --/omni.tbb.globalcontrol/maxThreadCount=12").app
try:
    from scripts.utils.helpers import make_env_policy
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    torch.manual_seed(base_cfg.seed)
    np.random.seed(base_cfg.seed)
    env, agent, _, _ = make_env_policy(base_cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager
    robot = cmd.asset
    ds = cmd.dataset
    policy = agent.get_rollout_policy("deploy")
    print(json.dumps({"policy_inputs": list(policy.in_keys), "target_joint_count": len(cmd.joint_idx_motion),
                      "dataset_joint_count": len(ds.joint_names), "all_target_joint_count": len(cmd.all_joint_idx_dataset),
                      "checkpoint": str(a.checkpoint.resolve())}), flush=True)

    cmd.zero_init_prob = 1.0
    for key in cmd.init_noise_params:
        cmd.init_noise_params[key] = 0.0
    # A few legacy snapshots omit this dataset flag; freeze the selected
    # references so reset cannot replace the manifest clip with a random one.
    ds.resample_on_reset = False
    ds.sample_once = True
    if a.source_dataset is None:
        for i, row in enumerate(rows):
            source = ds.datasets[paths.index(row["dataset"])]
            ci = int(row["clip"])
            start = int(source.starts[ci])
            end = start + int(row["frames"])
            idx = torch.arange(start, start + ds.max_step_size, device=source.starts.device).clamp_max(end - 1)
            ds._buf_A[i] = ds._to_float(source.data[idx].to(env.device), dtype=ds.buffer_float_dtype)
            ds._len_A[i] = int(row["frames"])
            for extra in ds.dataset_extra_keys:
                ds._info_A[extra["name"]][i] = source.info[extra["name"]][ci].to(env.device)
        # These buffers are the fixed protocol trajectories. Do not let a
        # legacy command reset resample over the explicitly populated rows.
        ds.reset = lambda env_ids: ds._len_A[env_ids]
        cmd.lengths[:] = ds._len_A
        cmd.t.zero_()
        cmd._motion = ds.get_slice(None, cmd.t, steps=cmd.future_steps)
    td = env.reset()
    for i, row in enumerate(rows):
        source = ds.datasets[paths.index(row["dataset"])]
        st = int(source.starts[int(row["clip"])])
        expected = (int(source.data.motion_id[st].item()), int(source.data.step[st].item()))
        got = (int(cmd._motion.motion_id[i, 0]), int(cmd._motion.step[i, 0]))
        if expected != got:
            raise RuntimeError(f"clip {i}: expected source motion/frame {expected}, got {got}")
        row["source_motion_id"], row["source_start_step"] = expected

    # Record both the joints fed to the native CEER policy and the complete
    # pose error over all dataset joints, making interface coverage explicit.
    input_joint_ids = torch.as_tensor(cmd.joint_idx_asset, device=env.device, dtype=torch.long)
    all_motion_ids = torch.as_tensor(cmd.all_joint_idx_dataset, device=env.device, dtype=torch.long)
    all_asset_ids = torch.as_tensor(cmd.all_joint_idx_asset, device=env.device, dtype=torch.long)
    n = len(rows)
    lengths = cmd.lengths.long()
    max_length = int(lengths.max().item())
    metric_sums = {k: torch.zeros(n, device=env.device) for k in
                   ("full_joint_mae_rad", "policy_joint_mae_rad", "root_xy_error_m", "root_z_error_m", "hand_position_error_m")}
    counts = torch.zeros(n, device=env.device)
    fall_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    hand_asset_ids = torch.tensor([robot.body_names.index(f"{s}_hand_mimic") for s in ("left", "right")], device=env.device)
    hand_motion_ids = torch.tensor([ds.body_names.index(f"{s}_hand_mimic") for s in ("left", "right")], device=env.device)
    video = None
    if make_video:
        if not hasattr(cmd, "force_apply_buffer"):
            cmd.force_apply_buffer = torch.zeros(n, len(robot.body_names), 3, device=env.device)
        if not hasattr(cmd, "position_apply_buffer"):
            cmd.position_apply_buffer = robot.data.body_pos_w.clone()
        from scripts.compliance_eval.visual import Video
        video = Video(env, cmd, a.output / "ceer_tracking_72.mp4", tracking=True)

    def accumulate(valid):
        validf = valid.float()
        q = robot.data.joint_pos
        metric_sums["full_joint_mae_rad"].add_((q[:, all_asset_ids] - cmd._motion.joint_pos[:, 0, all_motion_ids]).abs().mean(-1) * validf)
        metric_sums["policy_joint_mae_rad"].add_((q[:, input_joint_ids] - cmd._motion.joint_pos[:, 0, cmd.joint_idx_motion]).abs().mean(-1) * validf)
        root = robot.data.root_pos_w - env.scene.env_origins
        target_root = cmd._motion.root_pos_w[:, 0]
        metric_sums["root_xy_error_m"].add_((root[:, :2] - target_root[:, :2]).norm(dim=-1) * validf)
        metric_sums["root_z_error_m"].add_((root[:, 2] - target_root[:, 2]).abs() * validf)
        hands = robot.data.body_pos_w[:, hand_asset_ids] - env.scene.env_origins[:, None]
        target_hands = cmd._motion.body_pos_w[:, 0, hand_motion_ids]
        metric_sums["hand_position_error_m"].add_((hands - target_hands).norm(dim=-1).mean(-1) * validf)
        counts.add_(validf)
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        fallen = (root[:, 2] < .30) | (tilt > np.deg2rad(75)) | (~torch.isfinite(q).all(-1))
        newfall = (fall_step < 0) & fallen & valid
        fall_step[newfall] = cmd.t[newfall].long()

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        if video is not None:
            video.frame(0.0, "CEER LOW-LEVEL", False)
        accumulate(torch.ones(n, dtype=torch.bool, device=env.device))
        for frame in range(1, max_length):
            td = policy(td)
            td = env.step(td)["next"]
            accumulate(frame < lengths)
            if video is not None and frame % 2 == 0:
                video.frame(frame * env.step_dt, "CEER LOW-LEVEL", bool((fall_step[0] >= 0).item()))
            if not make_video and frame % 200 == 0:
                print(json.dumps({"frame": frame, "max_frames": max_length, "elapsed_s": round(time.time()-started, 1)}), flush=True)
    if video is not None:
        video.close()

    per_clip = []
    for i, row in enumerate(rows):
        d = max(float(counts[i]), 1.)
        per_clip.append({"split": row["split"], "dataset": row["dataset"], "clip": row["clip"],
                         "source_motion_id": row["source_motion_id"], "source_start_step": row["source_start_step"],
                         "category": row["category"], "frames": int(counts[i].item()),
                         **{k: float(v[i].item()/d) for k, v in metric_sums.items()},
                         "fall": bool(fall_step[i] >= 0),
                         "first_fall_frame": int(fall_step[i].item()) if fall_step[i] >= 0 else None})
    def summary(group):
        ks = list(metric_sums)
        if not group:
            return {"clips": 0, "frames": 0, "fall_clips": 0,
                    "clip_mean": {k: None for k in ks}, "frame_weighted_mean": {k: None for k in ks}}
        return {"clips": len(group), "frames": sum(c["frames"] for c in group),
                "fall_clips": sum(c["fall"] for c in group),
                "clip_mean": {k: float(np.mean([c[k] for c in group])) for k in ks},
                "frame_weighted_mean": {k: float(np.average([c[k] for c in group], weights=[c["frames"] for c in group])) for k in ks}}
    result = {"checkpoint": str(a.checkpoint.resolve()), "manifest_version": manifest["version"],
              "policy_inputs": list(policy.in_keys), "policy_target_joint_count": len(cmd.joint_idx_motion),
              "dataset_full_joint_count": len(all_motion_ids), "input_is_full_body": len(cmd.joint_idx_motion) == len(all_motion_ids),
              "physical_force": False, "dt_s": float(env.step_dt), "per_clip": per_clip,
              "all_160": summary(per_clip),
              "validation_128": summary([x for x in per_clip if x["split"] == "validation"]),
              "train_diagnostic_32": summary([x for x in per_clip if x["split"] == "train_diagnostic"]),
              "video": str(a.output / "ceer_tracking_72.mp4") if video is not None else None,
              "wall_seconds": time.time()-started}
    name = "ceer_tracking72.json" if make_video else "ceer_tracking_suite.json"
    (a.output / name).write_text(json.dumps(result, indent=2))
    print("CEER_TRACKING_COMPLETE", json.dumps({k: result[k] for k in ("policy_target_joint_count", "dataset_full_joint_count", "input_is_full_body", "validation_128", "all_160") if k in result}), flush=True)
    from isaaclab.sim import SimulationContext
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); _os._exit(1)
else:
    app.close()
