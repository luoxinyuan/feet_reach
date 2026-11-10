"""Replay the standard passive-whole-body tracking72 clip with native SONIC."""
import os
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--clip", type=int, default=79)
parser.add_argument("--output", type=Path, required=True)
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
OmegaConf.set_readonly(cfg, False)
cfg.checkpoint_path = str(args.checkpoint.resolve())
cfg.vecnorm = "eval"
cfg.seed = 20261108
cfg.task.num_envs = 1
cfg.task.max_episode_length = 2000
cfg.task.command.guidance_enabled = False
cfg.task.command.dataset.mem_paths = ["passive_whole_body_full/val"]
cfg.task.command.dataset.path_weights = [1.0]
cfg.task.command.dataset.sample_once = True
cfg.task.command.dataset.resample_on_reset = False
cfg.task.command.dataset.fix_ds = 0
cfg.task.command.dataset.fix_motion_id = args.clip
cfg.task.viewer.resolution = [960, 540]
cfg.task.reward = {}
cfg.task.termination = {}
cfg.task.randomization = {}
cfg.eval_render = True
cfg.headless = True
del state

app = AppLauncher(
    headless=True,
    device="cuda:0",
    enable_cameras=True,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=12 --/omni.tbb.globalcontrol/maxThreadCount=12",
).app

try:
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul
    from scripts.compliance_eval.visual import Video
    from scripts.utils.helpers import make_env_policy

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env, agent, _, _ = make_env_policy(cfg)
    env.eval()
    env.base_env.eval()
    agent.eval()

    cmd = env.command_manager
    robot = cmd.asset
    manager = env.action_manager
    if not hasattr(manager, "sonic") or not hasattr(manager, "history"):
        raise RuntimeError("Checkpoint did not construct the native Sonic action adapter")
    if len(manager.joint_ids) != 29:
        raise RuntimeError(f"Expected 29 Sonic joints, got {len(manager.joint_ids)}")

    # Start at the exact first pose of the fixed validation clip.
    cmd.zero_init_prob = 1.0
    for key in cmd.init_noise_params:
        cmd.init_noise_params[key] = 0.0
    td = env.reset()
    expected_source_motion_id = int(
        cmd.dataset.datasets[0].data.motion_id[cmd.dataset.datasets[0].starts[args.clip]].item()
    )
    if int(cmd.t[0]) != 0 or int(cmd._motion.motion_id[0, 0]) != expected_source_motion_id:
        raise RuntimeError(
            f"Expected validation clip {args.clip} (source motion {expected_source_motion_id}) at frame 0; got "
            f"motion={int(cmd._motion.motion_id[0, 0])}, frame={int(cmd.t[0])}"
        )

    sonic_joint_ids = torch.tensor(
        [cmd.dataset.joint_names.index(name) for name in manager.joint_names],
        device=env.device,
        dtype=torch.long,
    )
    original_forward = manager.sonic.forward
    reference_q = reference_qd = reference_orientation = None

    # SONIC receives the same 10-frame future joint and root-orientation window
    # used by its native tracking interface. The RL force-residual actor is not
    # used here: its action would alter a self-generated reference, not this clip.
    def forward_tracking(_q_reference, _orientation_6d, proprioception):
        return original_forward(
            reference_q,
            reference_orientation,
            proprioception,
            qd_reference=reference_qd,
        )

    manager.sonic.forward = forward_tracking
    length = int(cmd.lengths[0].item())
    steps = length - 1
    if steps <= 0:
        raise RuntimeError(f"Tracking clip too short: {length} frames")

    video_path = args.output / "sonic_tracking_72.mp4"
    renderer = Video(env, cmd, video_path, tracking=True)
    root_errors, joint_errors, ee_errors = [], [], []
    first_fall_s = None
    hand_names = ["left_hand_mimic", "right_hand_mimic"]
    hand_asset_ids = torch.tensor([robot.body_names.index(n) for n in hand_names], device=env.device)
    hand_motion_ids = torch.tensor([cmd.dataset.body_names.index(n) for n in hand_names], device=env.device)
    with torch.no_grad():
        # Match the existing tracking72 video cadence and framing: full clip,
        # 960x540, 25 fps, cyan translucent reference robot, fixed camera setup.
        renderer.frame(0.0, "SONIC (PRETRAINED)", False)
        for step in range(1, steps + 1):
            # env.step calls command.before_update before the action manager, so
            # the first action tracks frame 1, then advances one frame per 20 ms.
            start = torch.tensor([step], device=env.device, dtype=torch.long)
            future = cmd.dataset.get_slice(
                torch.zeros_like(start), start, steps=torch.arange(10, device=env.device)
            )
            reference_q = future.joint_pos[:, :, sonic_joint_ids]
            reference_qd = future.joint_vel[:, :, sonic_joint_ids]
            relative_orientation = quat_mul(
                quat_conjugate(robot.data.root_quat_w[:, None].expand(-1, 10, -1)),
                future.root_quat_w,
            )
            reference_orientation = matrix_from_quat(relative_orientation)[..., :, :2].reshape(1, 10, 6)

            # The compliance residual is intentionally zeroed; this is a pure
            # pretrained SONIC motion-tracking check.
            td["action"] = torch.zeros(1, manager.action_dim, device=env.device)
            td = env.step(td)["next"]

            target_root = cmd._motion.root_pos_w[:, 0] + env.scene.env_origins
            actual_root = robot.data.root_pos_w
            root_errors.append(float((actual_root[:, :2] - target_root[:, :2]).norm(dim=-1).mean()))
            joint_target = cmd._motion.joint_pos[:, 0, sonic_joint_ids]
            joint_actual = robot.data.joint_pos[:, manager.joint_ids]
            joint_errors.append(float((joint_actual - joint_target).abs().mean()))
            target_hands = cmd._motion.body_pos_w[:, 0, hand_motion_ids] + env.scene.env_origins[:, None]
            actual_hands = robot.data.body_pos_w[:, hand_asset_ids]
            ee_errors.append(float((actual_hands - target_hands).norm(dim=-1).mean()))
            height = float(actual_root[0, 2] - env.scene.env_origins[0, 2])
            tilt = float(torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
            if first_fall_s is None and (height < 0.45 or tilt > np.pi / 3):
                first_fall_s = (step + 1) * env.step_dt

            if step % 2 == 0:
                renderer.frame(step * env.step_dt, "SONIC (PRETRAINED)", first_fall_s is not None)
    renderer.close()
    report = {
        "controller": "pretrained frozen SONIC native tracking backend",
        "residual_student_used": False,
        "clip_index": args.clip,
        "source_motion_id": expected_source_motion_id,
        "frames": length,
        "duration_s": steps * env.step_dt,
        "video": str(video_path),
        "mean_root_xy_error_m": float(np.mean(root_errors)),
        "mean_joint_mae_rad": float(np.mean(joint_errors)),
        "mean_hand_position_error_m": float(np.mean(ee_errors)),
        "first_fall_s": first_fall_s,
        "force_guidance": False,
        "reference": "10-frame future full-body joint position/velocity plus relative root orientation",
    }
    report_path = args.output.parent / "metrics" / "sonic_tracking72.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print("SONIC_TRACKING72_COMPLETE", json.dumps(report), flush=True)
    SimulationContext.instance()._disable_app_control_on_stop_handle = True
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
