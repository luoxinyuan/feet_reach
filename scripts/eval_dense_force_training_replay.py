"""Replay one training clip and its cached 23-link forces with the dense student."""
import os
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--motion-id", type=int, default=7075)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher

checkpoint = args.checkpoint.resolve()
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
OmegaConf.set_struct(cfg, False)
cfg.checkpoint_path = str(checkpoint)
cfg.vecnorm = "eval"
cfg.seed = 20261028
cfg.task.num_envs = 1
cfg.task.max_episode_length = 1000
cfg.task.command.dataset.mem_paths = [str(ROOT / "dataset/passive_whole_body_full/train")]
cfg.task.command.dataset.path_weights = [1.0]
cfg.task.command.dataset.resample_on_reset = False
cfg.task.command.dataset.fix_ds = 0
cfg.task.command.dataset.fix_motion_id = args.motion_id
cfg.task.command.guidance_cache = str(ROOT / "artifacts/ceer_dense_link/full23/train")
cfg.task.command.guidance_enabled = True
cfg.wandb.mode = "disabled"
cfg.eval_render = True
cfg.headless = True
cfg.task.viewer.resolution = [1280, 720]
cfg.task.viewer.eye = [2.8, 0.0, 0.15]
cfg.task.viewer.lookat = [0.0, 0.0, 0.0]
del state

app = AppLauncher(
    headless=True, device="cuda:0", enable_cameras=True,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=16 --/omni.tbb.globalcontrol/maxThreadCount=16",
).app
try:
    import hashlib
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Usd, UsdGeom, Gf
    from isaacsim.core.utils.viewports import set_camera_view
    from isaaclab.sim import SimulationContext
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.utils.helpers import make_env_policy
    from active_adaptation.utils.math import quat_apply

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval(); env.base_env.eval(); agent.eval()
    cmd = env.command_manager
    robot = cmd.asset
    cmd.zero_init_prob = 1.0  # Start at the first frame so cached forces align with the clip.
    policy = agent.get_rollout_policy("deploy")
    if set(policy.in_keys) != {"policy", "estimator_history_"}:
        raise RuntimeError(f"Unexpected student inputs: {policy.in_keys}")
    td = env.reset()
    selected = int(cmd._motion.motion_id[0, 0])
    if int(cmd.cache_offsets[selected]) < 0:
        raise RuntimeError(f"Selected source motion {selected} has no cached force data")

    root = robot.data.root_pos_w[0].cpu().numpy().copy()
    target = root.copy()
    eye = root + np.array([2.8, 0.0, 0.15])
    env.base_env.cfg.viewer.eye = [2.8, 0.0, 0.15]
    env.base_env.cfg.viewer.lookat = [0.0, 0.0, 0.0]
    set_camera_view(eye.tolist(), target.tolist())
    stage = __import__("omni.usd", fromlist=["get_context"]).get_context().get_stage()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    writer = imageio.get_writer(args.output / "rollout.mp4", fps=25, codec="libx264", quality=8, macro_block_size=1)

    def project(world_point):
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/OmniverseKit_Persp")).GetCamera(Usd.TimeCode.Default())
        view_projection = camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix()
        v = Gf.Vec4d(float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0) * view_projection
        if abs(float(v[3])) < 1e-8:
            return np.array([-100.0, -100.0])
        return np.array([(float(v[0] / v[3]) * .5 + .5) * 1280,
                         (.5 - float(v[1] / v[3]) * .5) * 720])

    def draw_arrow(draw, start, end, color):
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm < 3.0:
            return
        direction = delta / norm
        normal = np.array([-direction[1], direction[0]])
        tip = end
        base = end - direction * 17.0
        draw.line([tuple(start), tuple(end)], fill=color, width=6)
        draw.polygon([tuple(tip), tuple(base + normal * 8), tuple(base - normal * 8)], fill=color)
        draw.ellipse((start[0]-5, start[1]-5, start[0]+5, start[1]+5), fill=color)

    point_names = list(cmd.point_names)
    colors = {"actual": (255, 154, 28), "estimated": (40, 225, 255)}
    max_steps = min(1000, int(cmd.lengths[0].item()))
    trace = []
    active_error = []
    actual_peak = 0.0
    fall_time = None
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for step in range(max_steps):
            td = policy(td)
            td = env.step(td)["next"]
            if step % 2 == 0 or step == max_steps - 1:
                root_now = robot.data.root_pos_w[0].cpu().numpy().copy()
                target_now = root_now.copy()
                eye_now = root_now + np.array([2.8, 0.0, 0.15])
                env.base_env.cfg.viewer.eye = [2.8, 0.0, 0.15]
                env.base_env.cfg.viewer.lookat = [0.0, 0.0, 0.0]
                set_camera_view(eye_now.tolist(), target_now.tolist())
                image = Image.fromarray(env.render("rgb_array"))
                draw = ImageDraw.Draw(image, "RGBA")

                force_world = cmd.point_force_buffer[0].detach().cpu().numpy()
                point_world = robot.data.body_pos_w[0, cmd.point_ids].detach().cpu().numpy()
                estimate_td = agent.estimator(td.clone())
                estimate_local = estimate_td["force_input_"].reshape(1, len(point_names), 3)[0] * 30.0
                estimate_world = quat_apply(
                    robot.data.root_quat_w[:, None].expand(-1, len(point_names), -1),
                    estimate_local[None],
                )[0].detach().cpu().numpy()

                active = np.flatnonzero(np.linalg.norm(force_world, axis=-1) > 0.25)
                actual_peak = max(actual_peak, float(np.linalg.norm(force_world, axis=-1).max()))
                active_names = []
                for slot in active:
                    actual = force_world[slot]
                    estimated = estimate_world[slot]
                    active_names.append(point_names[int(slot)])
                    actual_n = float(np.linalg.norm(actual))
                    if actual_n > .25:
                        start = project(point_world[slot])
                        end = project(point_world[slot] + actual / actual_n * min(.9, .045 * actual_n))
                        draw_arrow(draw, start, end, colors["actual"])
                    estimate_n = float(np.linalg.norm(estimated))
                    if estimate_n > .25:
                        # Slightly offset the cyan overlay so both vectors remain visible.
                        start3 = point_world[slot] + np.array([0.0, 0.0, .035])
                        end3 = start3 + estimated / estimate_n * min(.9, .045 * estimate_n)
                        draw_arrow(draw, project(start3), project(end3), colors["estimated"])
                    active_error.append(float(np.linalg.norm(estimated - actual)))

                t = (step + 1) * float(env.step_dt)
                draw.rectangle((0, 0, 1280, 108), fill=(8, 14, 23, 225))
                draw.text((18, 10), "23-POINT STUDENT • TRAINING CLIP REPLAY • FRONTAL VIEW", font=small, fill=(190, 225, 255))
                draw.text((18, 39), f"motion {selected}   |   {t:.2f} / {max_steps * env.step_dt:.1f} s   |   active links: {len(active)}", font=font, fill="white")
                names_text = ", ".join(active_names[:5]) + (" ..." if len(active_names) > 5 else "")
                draw.text((18, 76), names_text if names_text else "No applied force this frame", font=small, fill=(255, 220, 150))
                draw.rectangle((14, 650, 590, 706), fill=(8, 14, 23, 215))
                draw.text((26, 657), "Orange: cached applied force   Cyan: student estimator", font=small, fill="white")
                writer.append_data(np.asarray(image))

            forces = cmd.point_force_buffer[0].detach().cpu().numpy()
            trace.append({
                "time_s": (step + 1) * float(env.step_dt),
                "motion_id": selected,
                "dataset_step": int(cmd._motion.step[0, 0]),
                "active_links": [point_names[i] for i in np.flatnonzero(np.linalg.norm(forces, axis=-1) > .25)],
                "force_N_by_link": forces.tolist(),
            })
            height = float(robot.data.root_pos_w[0, 2])
            tilt = float(torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
            if fall_time is None and (height < .45 or tilt > np.pi / 3):
                fall_time = (step + 1) * float(env.step_dt)
    writer.close()

    np.savez_compressed(
        args.output / "force_trace.npz",
        time_s=np.asarray([row["time_s"] for row in trace]),
        force_N_by_link=np.asarray([row["force_N_by_link"] for row in trace]),
    )
    (args.output / "trace.json").write_text(json.dumps(trace))
    summary = {
        "checkpoint": str(checkpoint), "requested_dataset_local_id": args.motion_id,
        "motion_id": selected,
        "dataset": "passive_whole_body_full/train",
        "guidance_cache": "artifacts/ceer_dense_link/full23/train",
        "video": "rollout.mp4", "video_fps": 25,
        "duration_s": max_steps * float(env.step_dt), "frames": max_steps // 2 + 1,
        "point_names": point_names, "max_applied_force_per_point_N": actual_peak,
        "mean_active_link_estimator_error_N": float(np.mean(active_error)) if active_error else None,
        "first_fall_s": fall_time,
        "visualization": {"orange": "cached applied force", "cyan": "student estimator output"},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DENSE_TRAINING_REPLAY_COMPLETE", json.dumps(summary), flush=True)
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
