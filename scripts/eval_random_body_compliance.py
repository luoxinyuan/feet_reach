"""Record a short randomized body-force compliance rollout for a segmented policy."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20261028)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
OmegaConf.set_struct(cfg, False)
command_cfg = cfg.task.command
dense_mode = command_cfg._target_.split(".")[-1] == "DenseLinkComplianceCommand"
if command_cfg._target_.split(".")[-1] not in ("SegmentedJacobianComplianceCommand", "DenseLinkComplianceCommand"):
    raise ValueError(f"Unsupported command: {command_cfg._target_}")
cfg.checkpoint_path = str(args.checkpoint.resolve())
cfg.vecnorm = "eval"
cfg.task.num_envs = 1
cfg.seed = args.seed
cfg.wandb.mode = "disabled"
cfg.eval_render = True
cfg.headless = True
cfg.task.command.guidance_enabled = False
if dense_mode:
    cfg.task.command.dataset.mem_paths = [str(ROOT / "dataset/passive_whole_body_full/val")]
    cfg.task.command.dataset.path_weights = [1.0]
    cfg.task.command.guidance_cache = str(ROOT / "artifacts/ceer_dense_link/full23/val")
else:
    cfg.task.command.dataset.mem_paths = ["passive_whole_body_full/val"]
cfg.task.reward = {}
cfg.task.termination = {}
cfg.task.max_episode_length = 2000
cfg.task.observe_on_reset = True
cfg.task.viewer.resolution = [1280, 720]
cfg.task.viewer.eye = [3.0, 0.0, 1.15]
cfg.task.viewer.lookat = [0.0, 0.0, 0.8]
if dense_mode:
    dense_manifest = json.loads((ROOT / "artifacts/ceer_dense_link/full23/val/manifest.json").read_text())
    configured_points = [{"name": name, "body": name, "offset": [0.0, 0.0, 0.0]}
                         for name in dense_manifest["body_names"]]
else:
    configured_points = list(command_cfg.force_points)
force_names = [point["name"] for point in configured_points]
categories = {}
for category, candidates in {
    "root/torso": ("root", "torso_origin"),
    "left hand": ("left_ee", "left_hand"),
    "right hand": ("right_ee", "right_hand"),
}.items():
    if dense_mode:
        continue
    match = next((name for name in candidates if name in force_names), None)
    if match is None:
        raise ValueError(f"Policy has no shared test point for {category}: {force_names}")
    categories[category] = force_names.index(match)
if dense_mode:
    categories = {name: i for i, name in enumerate(force_names)}
del state

app = AppLauncher(
    headless=True,
    device="cuda:0",
    enable_cameras=True,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=16 --/plugins/omni.tbb.globalcontrol/maxThreadCount=16",
).app
try:
    import omni.usd
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    from isaaclab.sim import SimulationContext
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.utils.helpers import make_env_policy
    from active_adaptation.utils.body_guidance import slew_force
    from active_adaptation.utils.point_guidance import reduce_point_forces
    from active_adaptation.utils.math import quat_apply, quat_apply_inverse

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval()
    env.base_env.eval()
    agent.eval()
    cmd = env.command_manager
    cmd.zero_init_prob = 1.0
    robot = cmd.asset
    manager = env.action_manager
    device = env.device

    original_reset = cmd.reset
    def reset_standing(ids):
        original_reset(ids)
        root = robot.data.default_root_state[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        root[:, 3:7] = root.new_tensor([1., 0., 0., 0.])
        root[:, 7:] = 0
        robot.write_root_state_to_sim(root, env_ids=ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[ids],
            torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids)
        cmd.joint_pos_boot_protect[ids] = robot.data.default_joint_pos[ids]
        if hasattr(cmd, "force"):
            cmd.force[ids] = 0
        if hasattr(cmd, "desired"):
            cmd.desired[ids] = 0
        cmd.force_apply_buffer[ids] = 0
        cmd.torque_apply_buffer[ids] = 0
        cmd.point_force_buffer[ids] = 0
        torch.manual_seed(args.seed + 1)

    callbacks = env.base_env._reset_callbacks
    callbacks[:] = [reset_standing if cb == original_reset else cb for cb in callbacks]
    assert reset_standing in callbacks
    # Freeze the reference motion. DenseLink still advances its history tick so the
    # causal force estimator receives a fresh proprioceptive sample each control step.
    cmd.before_update = (lambda: setattr(cmd, "log_step", cmd.log_step + 1)) if dense_mode else (lambda: None)
    cmd.update = lambda: None

    labels = list(categories)
    order = (rng.choice(len(labels), size=3, replace=False).tolist() if dense_mode
             else rng.permutation(len(labels)).tolist())
    points = []
    palette = [(1.0, .34, .04), (1.0, .78, .05), (.95, .15, .62)]
    for index in order:
        category = labels[index]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        points.append({
            "category": category,
            "slot": categories[category],
            "body": configured_points[categories[category]]["body"],
            "offset": list(configured_points[categories[category]]["offset"]),
            "direction_world": direction.tolist(),
            "peak_N": float(rng.uniform(4.5, 7.0) if dense_mode else rng.uniform(9., 15.)),
            "start_s": 1.0 + len(points) * 2.25,
            "duration_s": 1.75,
        })
    (args.output / "force_schedule.json").write_text(json.dumps({"seed": args.seed, "points": points}, indent=2))

    desired = torch.zeros_like(cmd.point_force_buffer)
    clock_step = 0
    def external_force(substep):
        t = (clock_step + substep / env.decimation) * env.step_dt
        desired.zero_()
        for force in points:
            start = force["start_s"]
            end = start + force["duration_s"]
            if start <= t < end:
                local_t = t - start
                ramp = .25
                if local_t < ramp:
                    amplitude = .5 - .5 * np.cos(np.pi * local_t / ramp)
                elif local_t > force["duration_s"] - ramp:
                    amplitude = .5 - .5 * np.cos(np.pi * (force["duration_s"] - local_t) / ramp)
                else:
                    amplitude = 1.
                direction = torch.tensor(force["direction_world"], device=device, dtype=desired.dtype)
                desired[0, force["slot"]] = direction * (force["peak_N"] * amplitude)
                break
        cmd.point_force_buffer.copy_(slew_force(cmd.point_force_buffer, desired, env.physics_dt))
        if dense_mode:
            cmd.force_apply_buffer.zero_()
            cmd.torque_apply_buffer.zero_()
            cmd.force_apply_buffer[:, cmd.point_ids] = cmd.point_force_buffer
        else:
            body_ids = cmd.point_ids[None].expand(1, -1)
            force_w, moment_w = reduce_point_forces(
                cmd.point_force_buffer, cmd.point_positions_w(), body_ids, robot.data.body_pos_w)
            cmd.force_apply_buffer.copy_(force_w)
            cmd.torque_apply_buffer.copy_(moment_w)
        cmd.position_apply_buffer.copy_(robot.data.body_pos_w)
        robot.has_external_wrench = False

    original_step = cmd.step
    pre_callbacks = env.base_env._pre_step_callbacks
    pre_callbacks[:] = [external_force if cb == original_step else cb for cb in pre_callbacks]
    assert external_force in pre_callbacks

    stage = omni.usd.get_context().get_stage()
    arrows = []
    arrow_colors = [(1., .32, .03), (.05, .85, 1.)] if dense_mode else [(1., .32, .03)]
    for index, color in enumerate(arrow_colors):
        material_path = f"/World/RandomForce/Material{index}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        shaft = UsdGeom.Cylinder.Define(stage, f"/World/RandomForce/Arrow{index}/shaft")
        shaft.CreateAxisAttr("Z"); shaft.CreateRadiusAttr(.012)
        head = UsdGeom.Cone.Define(stage, f"/World/RandomForce/Arrow{index}/head")
        head.CreateAxisAttr("Z"); head.CreateRadiusAttr(.035)
        for shape in (shaft, head):
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
        arrows.append((shaft, shaft.AddTransformOp(), head, head.AddTransformOp()))

    def set_arrow(index, start, force):
        shaft, shaft_op, head, head_op = arrows[index]
        magnitude = float(np.linalg.norm(force))
        for shape in (shaft, head):
            UsdGeom.Imageable(shape).GetVisibilityAttr().Set("inherited" if magnitude > .05 else "invisible")
        if magnitude <= .05:
            return
        direction = force / magnitude
        length = (.055 if not dense_mode else .075) * magnitude
        tip = min(.09, length * .35)
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, direction))).GetQuat()
        shaft.CreateHeightAttr(max(.001, length - tip)); head.CreateHeightAttr(tip)
        for op, position in [
            (shaft_op, start + direction * (length - tip) / 2),
            (head_op, start + direction * (length - tip / 2)),
        ]:
            matrix = Gf.Matrix4d(1.); matrix.SetRotate(rotation)
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float, position))); op.Set(matrix)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    writer = imageio.get_writer(args.output / "rollout.mp4", fps=25, codec="libx264", quality=8, macro_block_size=1)
    policy = agent.get_rollout_policy("deploy")
    if dense_mode:
        assert set(policy.in_keys) == {"policy", "estimator_history_"}
    else:
        assert list(policy.in_keys) == ["policy"]
    td = env.reset()
    env.base_env.episode_length_buf.zero_()
    initial_hash = hashlib.sha256(
        robot.data.root_state_w.cpu().numpy().tobytes() + robot.data.joint_pos.cpu().numpy().tobytes()
    ).hexdigest()
    duration = 9.0
    steps = round(duration / env.step_dt)
    rows = []
    fall_time = None
    estimator_errors_active = []
    estimator_errors_all = []
    estimated_force_world = np.zeros_like(cmd.point_force_buffer[0].cpu().numpy())
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for step in range(steps + 1):
            clock_step = step
            if step:
                td = policy(td)
                td = env.step(td)["next"]
            root = robot.data.root_pos_w[0].cpu().numpy().copy() - env.scene.env_origins[0].cpu().numpy()
            if dense_mode:
                point_positions = robot.data.body_pos_w[0, cmd.point_ids].cpu().numpy()
            else:
                point_positions = cmd.point_positions_w()[0].cpu().numpy()
            point_forces = cmd.point_force_buffer[0].cpu().numpy()
            active_slots = np.flatnonzero(np.linalg.norm(point_forces, axis=-1) > .05)
            if dense_mode:
                estimate_td = agent.estimator(td.clone())
                estimate_local = estimate_td["force_input_"].reshape(1, len(cmd.point_names), 3)[0] * 30.0
                quat = robot.data.root_quat_w
                estimate_world = quat_apply(
                    quat[:, None].expand(-1, len(cmd.point_names), -1), estimate_local[None])[0]
                estimated_force_world = estimate_world.detach().cpu().numpy()
                actual_local = quat_apply_inverse(
                    quat[:, None].expand(-1, len(cmd.point_names), -1), cmd.point_force_buffer)[0]
                error = (estimate_local - actual_local).norm(dim=-1).detach().cpu().numpy()
                estimator_errors_all.extend(error.tolist())
                if active_slots.size:
                    estimator_errors_active.extend(error[active_slots].tolist())
            if active_slots.size:
                slot = int(active_slots[0])
                set_arrow(0, point_positions[slot], point_forces[slot])
                if dense_mode:
                    set_arrow(1, point_positions[slot] + np.array([0., 0., .06]), estimated_force_world[slot])
            else:
                set_arrow(0, np.zeros(3), np.zeros(3))
                if dense_mode:
                    set_arrow(1, np.zeros(3), np.zeros(3))
            height = float(robot.data.root_pos_w[0, 2])
            tilt = float(torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
            if fall_time is None and (height < .45 or tilt > np.pi / 3):
                fall_time = step * env.step_dt
            active = next((f for f in points if f["start_s"] <= step * env.step_dt < f["start_s"] + f["duration_s"]), None)
            rows.append({"time_s": step * env.step_dt, "root_pos_m": root.tolist(), "forces_N": point_forces.tolist()})
            if step % 2 == 0 or step == steps:
                center = robot.data.root_pos_w[0].cpu().numpy().copy()
                center[2] = .72
                actual_root = robot.data.root_pos_w[0].cpu().numpy()
                env.base_env.cfg.viewer.eye = (center + np.array([1.8, -2.2, 1.0]) - actual_root).tolist()
                env.base_env.cfg.viewer.lookat = (center - actual_root).tolist()
                image = Image.fromarray(env.render("rgb_array"))
                draw = ImageDraw.Draw(image, "RGBA")
                draw.rectangle((0, 0, 1280, 113), fill=(10, 16, 25, 225))
                task_name = cfg.task.name
                draw.text((16, 10), f"{task_name} | randomized body force | {step * env.step_dt:.2f} / {duration:.1f} s", font=font, fill="white")
                if active and dense_mode:
                    applied_n = float(np.linalg.norm(point_forces[active["slot"]]))
                    estimated_n = float(np.linalg.norm(estimated_force_world[active["slot"]]))
                    draw.text((16, 41), f"{active['body']} | applied {applied_n:.1f} N (orange) | estimated {estimated_n:.1f} N (cyan)", font=small, fill=(255, 215, 120))
                elif active:
                    draw.text((16, 41), f"{active['category']} | {np.linalg.norm(point_forces[active['slot']]):.1f} N actual | orange arrow = force at application point", font=small, fill=(255, 215, 120))
                else:
                    draw.text((16, 41), "NO FORCE | orange arrow appears at the applied point", font=small, fill="white")
                magnitude_text = "4.5-7 N" if dense_mode else "9-15 N"
                legend = "orange = applied force, cyan = estimator output" if dense_mode else "orange = applied force"
                draw.text((16, 70), f"Random body points/directions, {magnitude_text} | {legend}", font=small, fill=(130, 230, 255))
                if fall_time is not None:
                    draw.text((16, 680), f"PHYSICAL FALL at {fall_time:.2f} s", font=small, fill=(255, 80, 80))
                writer.append_data(np.asarray(image))
                if step in (0, 100, 250, 400):
                    image.save(args.output / f"frame_{step:04d}.png")
    writer.close()
    np.savez_compressed(args.output / "trace.npz", **{
        "time_s": np.asarray([row["time_s"] for row in rows]),
        "root_pos_m": np.asarray([row["root_pos_m"] for row in rows]),
        "forces_N": np.asarray([row["forces_N"] for row in rows]),
    })
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
        "initial_state_sha256": initial_hash, "categories": categories,
        "force_schedule": points, "video": "rollout.mp4", "video_fps": 25,
        "duration_s": duration, "first_physical_fall_s": fall_time,
        "root_drift_max_m": float(np.linalg.norm(np.asarray([r["root_pos_m"][:2] for r in rows]) - rows[0]["root_pos_m"][:2], axis=-1).max()),
        "estimated_force_rmse_N_all_points": float(np.sqrt(np.mean(np.square(estimator_errors_all)))) if dense_mode and estimator_errors_all else None,
        "estimated_force_rmse_N_active_points": float(np.sqrt(np.mean(np.square(estimator_errors_active)))) if dense_mode and estimator_errors_active else None,
        "note": "Three sequential random point-force pulses; dense-link finetune policy uses only causal force-estimator history at deployment. Ground-truth applied forces are used for physical load and scoring, not policy input." if dense_mode else "Three sequential randomized point-force pulses at the trained root/torso and left/right hand locations; force is applied at the configured point offset and exposed through the policy force observation.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print("RANDOM_BODY_COMPLIANCE_VIDEO_COMPLETE", json.dumps(report), flush=True)
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
