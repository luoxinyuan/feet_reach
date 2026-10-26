#!/usr/bin/env python3
"""Generate fixed-root, single-hand EE trajectories tracked by frozen CEER.

The opposite hand is held at its initial root-relative pose. Each episode stores
simulated robot state and the commanded wrist reference at 50 Hz, plus a small
quality report. This script only generates motion; force guidance is prepared
offline in a separate step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(Path(__file__).resolve().parents[1] / "dataset"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smooth_path(start: np.ndarray, waypoints: np.ndarray, frames: int) -> np.ndarray:
    anchors = np.concatenate([start[None], waypoints, start[None]], axis=0)
    lengths = np.full(len(anchors) - 1, frames // (len(anchors) - 1), dtype=int)
    lengths[: frames % (len(anchors) - 1)] += 1
    out = []
    for i, n in enumerate(lengths):
        u = np.arange(n, dtype=np.float32) / max(n, 1)
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        out.append(anchors[i][None] + s[:, None] * (anchors[i + 1] - anchors[i])[None])
    out.append(start[None])
    return np.concatenate(out, axis=0)[:frames]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--waypoints", type=int, default=3)
    p.add_argument("--hold-motion-id", type=int, default=51)
    p.add_argument("--apply-force-test", action="store_true",
                   help="Apply one smooth 5–20 N probe force to the active hand; use only for validation")
    args = p.parse_args()
    if args.count < 1 or args.envs < 1 or args.duration < 1.0 or args.waypoints < 1:
        p.error("count/envs/waypoints must be positive and duration must be at least 1 s")
    out = args.output.resolve()
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if list(raw_dir.glob("*.npz")):
        raise FileExistsError(f"Raw output already contains trajectories: {raw_dir}")

    from omegaconf import OmegaConf
    import hydra
    import active_adaptation.learning  # register the project policy classes
    from isaaclab.app import AppLauncher

    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=[
            "task=G1/hl/ee/G1_hl_ee_compliance_pos_delta_force_b_student",
            f"task.num_envs={args.envs}", "wandb.mode=disabled",
            f"task.action.low_policy.checkpoint_path={ROOT}/.cache/ceer/checkpoint_final.pt",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = None
    cfg.seed = args.seed
    cfg.task.max_episode_length = max(100000, int(args.duration * 50 + 100))
    cfg.task.termination = {}
    cfg.task.eval_contact_forces = False
    # Start every rollout from an ordinary, near-symmetric standing hold pose.
    # This avoids carrying unrelated whole-body motion into the synthetic clip.
    cfg.task.command.dataset.mem_paths = ["passive_ee_continuous/train/hold"]
    cfg.task.command.dataset.path_weights = [1.0]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = args.hold_motion_id
    cfg.task.command.student_train = True
    cfg.task.command.disable_motion_finish = True
    cfg.headless = True
    cfg.app = {"headless": True, "enable_cameras": False}
    OmegaConf.resolve(cfg)

    app = AppLauncher(headless=True, device=args.device).app
    records: list[dict] = []
    try:
        from active_adaptation.envs import SimpleEnv
        from active_adaptation.utils.motion import MotionDataset
        from tensordict import TensorDict

        # The reference sample pool is drawn from ordinary AMASS hand poses.
        pool_ds = MotionDataset.create_from_path_lazy(
            "passive_ee_continuous/train/move", device=torch.device("cpu"))
        pool_ids = [pool_ds.body_names.index("left_hand_mimic"),
                    pool_ds.body_names.index("right_hand_mimic")]
        pools = [pool_ds.data.body_pos_b[:, i].float().numpy() for i in pool_ids]
        # Stay in the corresponding arm's natural half of the torso frame.
        # Restrict to the upright, forward workspace shared by ordinary
        # standing manipulation. The empirical pool also contains sitting and
        # behind-the-torso poses that are unsuitable for a standing EE probe.
        pools[0] = pools[0][(pools[0][:, 1] > 0.08) & (pools[0][:, 0] > 0.10)
                            & (pools[0][:, 2] > -0.12) & (pools[0][:, 2] < 0.32)]
        pools[1] = pools[1][(pools[1][:, 1] < -0.08) & (pools[1][:, 0] > 0.10)
                            & (pools[1][:, 2] > -0.12) & (pools[1][:, 2] < 0.32)]
        if min(map(len, pools)) < 100:
            raise RuntimeError("Insufficient ordinary-motion samples for both hands")

        base = SimpleEnv(cfg.task)
        base.eval()
        cmd = base.command_manager
        act = base.action_manager
        if not callable(getattr(act, "preload_low_policy", None)):
            raise RuntimeError("Configured action manager does not expose the frozen CEER policy")
        if getattr(act, "low_policy", None) is None:
            act.preload_low_policy()
        device = base.device
        fps = round(1.0 / base.step_dt)
        if fps != 50:
            raise RuntimeError(f"Expected 50 Hz control, found {fps} Hz")
        nframes = round(args.duration * fps)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        joint_names = list(cmd.asset.joint_names)
        body_names = list(cmd.asset.body_names)
        hand_ids = [body_names.index("left_hand_mimic"), body_names.index("right_hand_mimic")]
        total_batches = (args.count + args.envs - 1) // args.envs

        for batch in range(total_batches):
            start = batch * args.envs
            actual_count = min(args.envs, args.count - start)
            base.reset()
            base.episode_length_buf.zero_()
            # Construct references from the measured reset pose so the initial
            # command is continuous with the robot's actual hand transforms.
            root_p = cmd.asset.data.root_pos_w.detach().cpu().numpy()
            root_q_wxyz = cmd.asset.data.root_quat_w.detach().cpu().numpy()
            hand_p = cmd.asset.data.body_pos_w[:, hand_ids].detach().cpu().numpy()
            hand_q_wxyz = cmd.asset.data.body_quat_w[:, hand_ids].detach().cpu().numpy()
            ref0 = np.zeros((args.envs, 12), dtype=np.float32)
            for e in range(args.envs):
                rr = Rotation.from_quat(root_q_wxyz[e, [1, 2, 3, 0]])
                for h in range(2):
                    ref0[e, 3*h:3*h+3] = rr.inv().apply(hand_p[e, h] - root_p[e])
                    rq = Rotation.from_quat(hand_q_wxyz[e, h, [1, 2, 3, 0]])
                    ref0[e, 6+3*h:9+3*h] = (rr.inv() * rq).as_rotvec()

            refs = np.repeat(ref0[:, None, :], nframes, axis=1)
            sides = np.array([(args.start_index + start + e) % 2 for e in range(args.envs)], dtype=np.int64)
            side_names = ["left" if x == 0 else "right" for x in sides]
            for e in range(args.envs):
                active = int(sides[e])
                initial = ref0[e, 3*active:3*active+3]
                pool = pools[active]
                # Prefer waypoints with useful excursion while staying inside
                # the empirical reachable-pose cloud.
                chosen = []
                for _ in range(args.waypoints):
                    for _try in range(100):
                        candidate = pool[rng.integers(len(pool))]
                        distance = np.linalg.norm(candidate - initial)
                        if 0.08 <= distance <= 0.30:
                            chosen.append(candidate)
                            break
                    else:
                        chosen.append(pool[rng.integers(len(pool))])
                path = smooth_path(initial, np.asarray(chosen), nframes)
                refs[e, :, 3*active:3*active+3] = path

            target_t = torch.as_tensor(refs, device=device)
            init_root = cmd.asset.data.root_pos_w.clone()
            init_hand_rel = torch.as_tensor(ref0[:, :6].reshape(args.envs, 2, 3), device=device)
            cmd.set_root_and_wrist_6d_reference_override(target_t[:, 0])
            if args.apply_force_test:
                from active_adaptation.utils.math import quat_apply
                directions = rng.normal(size=(args.envs, 3)).astype(np.float32)
                directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-6)
                magnitudes = rng.uniform(5., 20., args.envs).astype(np.float32)
                ramp_frames = rng.integers(18, 27, args.envs)
                starts = rng.integers(15, 36, args.envs)
                holds = rng.integers(30, 66, args.envs)
                force_profile = np.zeros((nframes, args.envs), dtype=np.float32)
                for e in range(args.envs):
                    rise = min(nframes, int(starts[e] + ramp_frames[e]))
                    u = np.arange(rise-int(starts[e]), dtype=np.float32) / ramp_frames[e]
                    force_profile[int(starts[e]):rise, e] = magnitudes[e] * (10*u**3-15*u**4+6*u**5)
                    stop = min(nframes-1, rise+int(holds[e]))
                    force_profile[rise:stop, e] = magnitudes[e]
                    fall = min(nframes, stop+int(ramp_frames[e]))
                    u = 1.-np.arange(fall-stop, dtype=np.float32)/ramp_frames[e]
                    force_profile[stop:fall, e] = magnitudes[e]*(10*u**3-15*u**4+6*u**5)
                probe_state = {"step": 0}
                row_ids = torch.arange(args.envs, device=device)
                side_ids = torch.as_tensor(sides, device=device)
                body_ids = torch.where(side_ids == 0, hand_ids[0], hand_ids[1])
                direction_t = torch.as_tensor(directions, device=device)
                profile_t = torch.as_tensor(force_profile, device=device)
                def apply_probe_force(substep):
                    cmd.force_apply_buffer.zero_(); cmd.torque_apply_buffer.zero_()
                    q = cmd.asset.data.root_quat_w
                    force_w = quat_apply(q, direction_t) * profile_t[probe_state["step"]][:, None]
                    cmd.force_apply_buffer[row_ids, body_ids] = force_w
                    cmd.asset.has_external_wrench = False
                    cmd.force_apply_world = True
                base._pre_step_callbacks.append(apply_probe_force)
            # Capture actual robot states after each control frame.
            states = {key: [] for key in ("qpos", "qvel", "xpos", "xquat", "root_xyz_target", "hand_xyz_actual_b")}
            track = {"active_error": [], "fixed_error": [], "root_drift": [], "root_height": [], "tilt": []}
            with torch.no_grad():
                for t in range(nframes):
                    if args.apply_force_test:
                        probe_state["step"] = t
                    cmd.set_root_and_wrist_6d_reference_override(target_t[:, t])
                    td = TensorDict({}, batch_size=[args.envs], device=device)
                    base._compute_observation(td)
                    td["action"] = torch.zeros((args.envs, act.action_dim), device=device)
                    td["is_init"] = torch.zeros((args.envs, 1), dtype=torch.bool, device=device)
                    base.step(td)

                    root_pos = cmd.asset.data.root_pos_w
                    root_quat = cmd.asset.data.root_quat_w
                    q = cmd.asset.data.joint_pos
                    qv = cmd.asset.data.joint_vel
                    bp = cmd.asset.data.body_pos_w
                    bq = cmd.asset.data.body_quat_w
                    # Use quaternion inverse rotation for root-relative hand coordinates.
                    from active_adaptation.utils.math import quat_apply_inverse
                    root_rep = root_quat[:, None, :].expand(-1, 2, -1)
                    hand_rel = quat_apply_inverse(root_rep, bp[:, hand_ids] - root_pos[:, None])
                    qvel_root = torch.cat((cmd.asset.data.root_lin_vel_w, cmd.asset.data.root_ang_vel_w), dim=-1)
                    qpos = torch.cat((root_pos, root_quat, q), dim=-1)
                    qvel = torch.cat((qvel_root, qv), dim=-1)
                    for key, value in (("qpos", qpos), ("qvel", qvel), ("xpos", bp), ("xquat", bq),
                                       ("root_xyz_target", target_t[:, t, :3].reshape(args.envs, 1, 3)),
                                       ("hand_xyz_actual_b", hand_rel)):
                        states[key].append(value.detach().cpu().numpy().astype(np.float32))
                    root_err = torch.linalg.vector_norm(root_pos - init_root, dim=-1)
                    active_err = torch.linalg.vector_norm(hand_rel - target_t[:, t, :6].reshape(args.envs, 2, 3), dim=-1)
                    active_err = active_err.gather(1, torch.as_tensor(sides, device=device)[:, None]).squeeze(1)
                    fixed_err = active_err.new_zeros(args.envs)
                    for e in range(args.envs):
                        fixed_err[e] = torch.linalg.vector_norm(hand_rel[e, 1-int(sides[e])] - init_hand_rel[e, 1-int(sides[e])])
                    tilt = cmd.asset.data.projected_gravity_b[:, 2]
                    track["active_error"].append(active_err.detach().cpu().numpy())
                    track["fixed_error"].append(fixed_err.detach().cpu().numpy())
                    track["root_drift"].append(root_err.detach().cpu().numpy())
                    track["root_height"].append(root_pos[:, 2].detach().cpu().numpy())
                    track["tilt"].append(tilt.detach().cpu().numpy())
            if args.apply_force_test:
                base._pre_step_callbacks.remove(apply_probe_force)

            # Do not write partial final-batch lanes.
            arrays = {k: np.stack(v, axis=1)[:actual_count] for k, v in states.items()}
            metrics = {k: np.stack(v, axis=1)[:actual_count] for k, v in track.items()}
            for e in range(actual_count):
                idx = args.start_index + start + e
                active = int(sides[e])
                qpos = arrays["qpos"][e]
                qvel = arrays["qvel"][e]
                xpos = arrays["xpos"][e]
                xquat = arrays["xquat"][e]
                active_mae = float(metrics["active_error"][e].mean())
                fixed_mae = float(metrics["fixed_error"][e].mean())
                max_fixed = float(metrics["fixed_error"][e].max())
                max_root_drift = float(metrics["root_drift"][e].max())
                min_height = float(metrics["root_height"][e].min())
                max_tilt = float(np.arccos(np.clip(-metrics["tilt"][e], -1, 1)).max())
                stable = (min_height > 0.60 and max_tilt < 0.8 and max_root_drift < 0.20)
                track_ok = active_mae < 0.08 and fixed_mae < 0.05 and max_fixed < 0.12
                name = f"traj_{idx:06d}_{side_names[e]}"
                root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
                root_rotation = Rotation.from_quat(root_rot_xyzw)
                local_body_pos = np.einsum("tij,tbj->tbi", root_rotation.inv().as_matrix(),
                    xpos - qpos[:, None, :3]).astype(np.float32)
                body_rot_xyzw = xquat[..., [1, 2, 3, 0]].reshape(-1, 4)
                body_rotation_m = Rotation.from_quat(body_rot_xyzw).as_matrix().reshape(
                    nframes, len(body_names), 3, 3)
                root_rotation_inv_m = root_rotation.inv().as_matrix()
                local_rotation_m = np.einsum("tij,tbjk->tbik", root_rotation_inv_m, body_rotation_m)
                local_body_rot = Rotation.from_matrix(local_rotation_m.reshape(-1, 3, 3)).as_quat().reshape(
                    nframes, len(body_names), 4).astype(np.float32)
                np.savez_compressed(raw_dir / f"{name}.npz",
                    fps=np.int32(fps), qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat,
                    root_pos=qpos[:, :3].astype(np.float32), root_rot=root_rot_xyzw.astype(np.float32),
                    dof_pos=qpos[:, 7:].astype(np.float32), local_body_pos=local_body_pos,
                    local_body_rot=local_body_rot,
                    joint_names=np.asarray(joint_names, dtype=object), body_names=np.asarray(body_names, dtype=object),
                    target_root_wrist_6d=refs[e], active_side=np.asarray(side_names[e]),
                    quality=np.asarray([active_mae, fixed_mae, max_fixed, max_root_drift, min_height, max_tilt], dtype=np.float32))
                result = dict(index=idx, active_side=side_names[e], active_hand_mae_m=active_mae,
                    fixed_hand_mae_m=fixed_mae, fixed_hand_max_error_m=max_fixed,
                    max_root_drift_m=max_root_drift, min_root_height_m=min_height,
                    max_tilt_rad=max_tilt, stable=stable, tracking_ok=track_ok,
                    accepted=stable and track_ok, file=f"raw/{name}.npz")
                records.append(result)
            (out / "progress.json").write_text(json.dumps(dict(completed=len(records), total=args.count,
                accepted=sum(x["accepted"] for x in records), records=records), indent=2))
            print("GEN_BATCH", batch + 1, "/", total_batches, "completed", len(records),
                  "accepted", sum(x["accepted"] for x in records), flush=True)

        by_side = {}
        for side in ("left", "right"):
            subset = [r for r in records if r["active_side"] == side]
            by_side[side] = dict(count=len(subset), accepted=sum(r["accepted"] for r in subset),
                stable_fraction=float(np.mean([r["stable"] for r in subset])) if subset else 0.0,
                tracking_ok_fraction=float(np.mean([r["tracking_ok"] for r in subset])) if subset else 0.0,
                active_hand_mae_m=float(np.mean([r["active_hand_mae_m"] for r in subset])) if subset else None,
                fixed_hand_mae_m=float(np.mean([r["fixed_hand_mae_m"] for r in subset])) if subset else None,
                max_root_drift_m=float(np.mean([r["max_root_drift_m"] for r in subset])) if subset else None)
        report = dict(seed=args.seed, count=args.count, envs=args.envs, fps=fps, duration_s=args.duration,
            waypoints=args.waypoints, controller="frozen CEER low-level, zero high-level residual",
            reference_pool="passive_ee_continuous/train/move", acceptance=dict(stable="height>0.60m, tilt<0.8rad, root drift<0.20m",
                tracking="active EE mean error<0.08m, fixed EE mean<0.05m and max<0.12m"),
            by_side=by_side, accepted=sum(r["accepted"] for r in records), records=records)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        print("GENERATION_COMPLETE", json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        base.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
