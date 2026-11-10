#!/usr/bin/env python3
"""Run released full-BONES SuperMimic on feet_reach's fixed tracking clips.

Runs in the compatible supermimic_force_transition venv. The report preserves
the feet_reach clip set and error metrics, while explicitly identifying the
checkpoint-native MuJoCo Warp backend.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SM_ROOT = Path("/home/xl521/supermimic_force_transition")
sys.path.insert(0, str(SM_ROOT / "code"))

from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import heading_mat, mtv, quat_to_mat
from supermimic.sim_env import EnvCfg, TrackingSimEnv
from supermimic.task import GOAL_FRAME_DIM, NJ, TaskCfg


class ReleasedPolicy(nn.Module):
    def __init__(self, state):
        super().__init__()
        p = state["policy"]
        self.encoder = nn.ModuleList([
            nn.Linear(680, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512),
            nn.Linear(512, 256), nn.Linear(256, 32),
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(962, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 512),
            nn.Linear(512, 256), nn.Linear(256, 29),
        ])
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
        x = torch.cat([z, obs * self.obs_scale], dim=-1)
        for layer in self.decoder[:-1]:
            x = torch.nn.functional.silu(layer(x))
        return self.decoder[-1](x)


def make_goal(lib, frame, starts, lengths, q0, device):
    offsets = torch.arange(10, device=device)
    idx = starts[:, None] + frame[:, None] + offsets[None]
    idx = torch.minimum(idx, (starts + lengths - 1)[:, None])
    ref = lib.frame(idx)
    gravity_w = torch.zeros_like(ref["v"])
    gravity_w[..., 2] = -1.0
    gravity_body = mtv(ref["rot"], gravity_w)
    ang_body = mtv(ref["rot"], ref["w"])
    lin_heading = mtv(heading_mat(ref["rot"]), ref["v"])
    goal = torch.cat([
        ref["q"] - q0,
        ref["qd"],
        gravity_body,
        ang_body,
        lin_heading,
        ref["pos"][..., 2:3],
    ], dim=-1)
    if goal.shape[-1] != 68:
        raise RuntimeError(f"Expected 68 goal features per frame, got {goal.shape[-1]}")
    return goal.flatten(1)


def protocol_fall(state, ref):
    # Match the checkpoint's termination semantics: compare pelvis height and
    # projected-gravity tilt against the motion reference. Absolute pelvis
    # height is not a fall criterion here because valid crouch motions can put
    # the pelvis below 0.30 m.
    bad_height = (state["pos"][:, 2] - ref["pos"][:, 2]).abs() > 0.25
    bad_tilt = (state["rot"][:, 2, 2] - ref["rot"][:, 2, 2]).abs() > 0.8
    invalid = ~torch.isfinite(state["q"]).all(-1)
    return bad_height | bad_tilt | invalid


def flatten_policy_history(obs_hist: torch.Tensor) -> torch.Tensor:
    """Match mjlab's per-observation-term history layout used to train SuperMimic.

    ``obs_hist`` is stored as (N, H, 93), with one full observation frame per
    row. The checkpoint expects each term's history flattened independently,
    then the terms concatenated in this order: joint_pos, joint_vel,
    base_ang_vel, projected_gravity, actions. A plain transpose would instead
    make each scalar feature's history contiguous, which is a different input.
    """
    term_dims = (29, 29, 3, 3, 29)
    if obs_hist.ndim != 3 or obs_hist.shape[-1] != sum(term_dims):
        raise ValueError(f"Expected observation history (N,H,93), got {tuple(obs_hist.shape)}")
    chunks = []
    start = 0
    for width in term_dims:
        chunks.append(obs_hist[..., start : start + width].flatten(1))
        start += width
    return torch.cat(chunks, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--motions", type=Path, required=True)
    ap.add_argument("--motion-meta", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    policy = ReleasedPolicy(ck).to(device).eval()
    deploy = json.loads((a.checkpoint.parent / "deploy_config.json").read_text())
    lib = MotionLibrary(str(a.motions), str(device), bin_seconds=1.0)
    if lib.num_clips != len(json.loads(a.motion_meta.read_text())["clips"]):
        raise RuntimeError("Motion library and source protocol manifest differ in clip count")

    env_cfg = EnvCfg(
        command_interface="full", num_envs=lib.num_clips, episode_length_s=1.0e6,
        domain_rand=False, obs_noise=False, reset_noise=False, compile=False,
        eval_friction=1.0, nconmax=35, njmax=250,
    )
    task_cfg = TaskCfg(term_z=1.0e6, term_tilt=math.pi, noise_q=0.0, noise_qd=0.0,
                      noise_gyro=0.0, noise_grav=0.0)
    env = TrackingSimEnv(lib, env_cfg, task_cfg, str(device))
    env.eval_mode = True
    names = deploy["joint_names"]
    if env.joint_names != names:
        raise RuntimeError("Released checkpoint joint order does not match the compatible simulator")
    for name, actual, expected in (
        ("default_joint_pos", env.q0, torch.tensor(deploy["default_joint_pos"], device=device)),
        ("action_scale", env.action_scale, torch.tensor(deploy["action_scale"], device=device)),
    ):
        maxdiff = float((actual - expected).abs().max().item())
        if maxdiff > 1e-4:
            raise RuntimeError(f"Checkpoint {name} mismatch with simulator (max abs diff {maxdiff:.6g})")

    n = lib.num_clips
    ids = torch.arange(n, device=device)
    starts, lengths = lib.starts.clone(), lib.lengths.clone()
    env.reset()
    env.reset_envs(ids, frames=starts)
    env._finish_resets(ids, force=True)
    frame = torch.zeros(n, dtype=torch.long, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    fallen = torch.zeros(n, dtype=torch.bool, device=device)
    first_fall = torch.full((n,), -1, dtype=torch.long, device=device)
    counts = torch.zeros(n, dtype=torch.long, device=device)
    pre_fall_counts = torch.zeros(n, dtype=torch.long, device=device)
    metrics = {k: torch.zeros(n, device=device) for k in
               ("full_joint_mae_rad", "root_xy_error_m", "root_z_error_m", "hand_position_error_m")}
    pre_fall_metrics = {k: torch.zeros(n, device=device) for k in metrics}
    wrist_ids = torch.tensor(env.kin.body_index(["left_wrist_yaw_link", "right_wrist_yaw_link"]), device=device)
    wrist_to_hand = torch.tensor([0.116, 0.0, 0.0], device=device)
    initial_ref = lib.frame(starts)
    initial_state = env.read_state()
    initial_alignment = {
        "joint_max_abs_rad": float((initial_state["q"] - initial_ref["q"]).abs().max().item()),
        "root_position_max_abs_m": float((initial_state["pos"] - initial_ref["pos"]).abs().max().item()),
    }

    def record(s, at_frame, valid):
        if not valid.any():
            return
        ref_frame = torch.minimum(at_frame, lengths - 1)
        ref = lib.frame(starts + ref_frame)
        qerr = (s["q"] - ref["q"]).abs().mean(-1)
        root = s["pos"]
        root_xy = (root[:, :2] - ref["pos"][:, :2]).norm(dim=-1)
        root_z = (root[:, 2] - ref["pos"][:, 2]).abs()
        P, R, _, _ = env.kin.forward(s["pos"], s["rot"], s["q"], s["v"], s["w"], s["qd"])
        rP, rR, _, _ = env.kin.forward(ref["pos"], ref["rot"], ref["q"], ref["v"], ref["w"], ref["qd"])
        hand_pos = P[:, wrist_ids] + (R[:, wrist_ids] @ wrist_to_hand).squeeze(-1)
        ref_hand_pos = rP[:, wrist_ids] + (rR[:, wrist_ids] @ wrist_to_hand).squeeze(-1)
        hand = (hand_pos - ref_hand_pos).norm(dim=-1).mean(-1)
        values = (("full_joint_mae_rad", qerr), ("root_xy_error_m", root_xy),
                  ("root_z_error_m", root_z), ("hand_position_error_m", hand))
        before_fall = valid & ~fallen & ~protocol_fall(s, ref)
        for key, value in values:
            metrics[key].add_(value * valid)
            pre_fall_metrics[key].add_(value * before_fall)
        counts.add_(valid.long())
        pre_fall_counts.add_(before_fall.long())
        now_fallen = protocol_fall(s, ref) & valid
        new = now_fallen & ~fallen
        first_fall[new] = at_frame[new]
        fallen.logical_or_(now_fallen)

    state = env.read_state()
    max_frames = int(lengths.max().item())
    with torch.inference_mode():
        for step in range(max_frames):
            record(state, frame, active)
            if not active.any():
                break
            raw_hist = flatten_policy_history(env.obs_hist)
            # TrackingSimEnv.ref_future() for the full-motion interface starts
            # one frame ahead of the current simulator state. Match that timing
            # (also used by the feet_reach tracking-suite evaluator).
            goal = make_goal(lib, frame + 1, starts, lengths, env.q0, device)
            action = policy(raw_hist, goal)
            action = torch.where(active[:, None], action, torch.zeros_like(action))
            state_next, info = env.step(action)
            next_frame = frame + 1
            last_frame = next_frame >= lengths - 1
            ending = active & (info["done"] | last_frame)
            record(state_next, next_frame, ending)
            active &= ~ending
            frame = next_frame
            state = state_next
            if (step + 1) % 100 == 0:
                print(json.dumps({"steps": step + 1, "max_steps": max_frames,
                                  "finished_clips": int((~active).sum().item()),
                                  "elapsed_s": round(time.time() - started, 1)}), flush=True)

    per_clip = []
    meta = json.loads(a.motion_meta.read_text())
    for i, row in enumerate(meta["clips"]):
        denom = max(int(counts[i].item()), 1)
        per_clip.append({
            "split": row["split"], "dataset": row["dataset"], "clip": int(row["clip"]),
            "category": row["category"], "frames": int(counts[i].item()),
            "pre_fall_frames": int(pre_fall_counts[i].item()),
            **{k: float(v[i].item() / denom) for k, v in metrics.items()},
            **{f"pre_fall_{k}": float(pre_fall_metrics[k][i].item() / max(int(pre_fall_counts[i].item()), 1)) for k in metrics},
            "fall": bool(fallen[i].item()),
            "first_fall_frame": int(first_fall[i].item()) if first_fall[i] >= 0 else None,
        })

    def summary(rows):
        if not rows:
            return {"clips": 0, "frames": 0, "fall_clips": 0}
        return {
            "clips": len(rows), "frames": sum(x["frames"] for x in rows),
            "pre_fall_frames": sum(x["pre_fall_frames"] for x in rows),
            "fall_clips": sum(x["fall"] for x in rows),
            "clip_mean": {k: float(sum(x[k] for x in rows) / len(rows)) for k in metrics},
            "pre_fall_clip_mean": {k: float(sum(x[f"pre_fall_{k}"] for x in rows) / len(rows)) for k in metrics},
            "frame_weighted_mean": {
                k: float(sum(x[k] * x["frames"] for x in rows) / max(sum(x["frames"] for x in rows), 1))
                for k in metrics
            },
            "pre_fall_frame_weighted_mean": {
                k: float(sum(x[f"pre_fall_{k}"] * x["pre_fall_frames"] for x in rows) /
                           max(sum(x["pre_fall_frames"] for x in rows), 1))
                for k in metrics
            },
        }

    result = {
        "checkpoint": str(a.checkpoint.resolve()), "checkpoint_iteration": int(ck["iteration"]),
        "protocol": meta["manifest_version"], "controller": "released SuperMimic BONES-SEED base policy",
        "simulator_backend": "checkpoint-compatible mjlab / MuJoCo Warp (not IsaacLab)",
        "clip_count": n, "dt_s": 0.02, "physical_force": False,
        "fall_definition": "abs(root_z - ref_root_z) > 0.25 m or abs(projected_gravity_z - ref_projected_gravity_z) > 0.8 or nonfinite joint state",
        "observation_layout": "per-term history flattened and concatenated: joint_pos, joint_vel, base_ang_vel, projected_gravity, actions",
        "initial_state_alignment": initial_alignment,
        "goal_timing": "10 reference frames starting at the next simulator frame (t+1)",
        "hand_metric": "wrist_yaw_link FK plus documented 0.116 m local-X hand offset",
        "all_160": summary(per_clip),
        "validation_128": summary([x for x in per_clip if x["split"] == "validation"]),
        "train_diagnostic_32": summary([x for x in per_clip if x["split"] == "train_diagnostic"]),
        "per_clip": per_clip, "wall_seconds": time.time() - started,
    }
    out = a.output / "supermimic_tracking_suite.json"
    out.write_text(json.dumps(result, indent=2))
    print("SUPERMIMIC_TRACKING_COMPLETE", json.dumps({k: result[k] for k in
          ("all_160", "validation_128", "train_diagnostic_32", "wall_seconds")}), flush=True)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
