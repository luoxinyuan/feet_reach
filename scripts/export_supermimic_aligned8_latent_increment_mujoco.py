#!/usr/bin/env python3
"""Export the aligned-8 SuperMimic latent-increment curriculum policy to MuJoCo."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
TRAIN = ROOT / "outputs/20261125_194909-supermimic-aligned8-online-latent-increment-contact-spring-gpu23/train"
CHECKPOINT = TRAIN / "wandb/run-20261125_195009-xzks5hv0/files/checkpoint_final.pt"
SM_CHECKPOINT = TRAIN / "supermimic_frozen_checkpoint.pt"
DEPLOY_CONFIG = TRAIN / "supermimic_deploy_config.json"
FORCE_MANIFEST = TRAIN / "force_cache_manifest.json"
BUNDLE = ROOT / "deploy/mujoco_whole_body/20261126-supermimic-aligned8-contact-spring-latent-increment"
TEMPLATE = ROOT / "deploy/mujoco_whole_body/20261118-supermimic-3kp-jacobian-latent-increment-curriculum/metadata.json"


class Actor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(obs_dim, 512), nn.LayerNorm(512), nn.Mish(),
                                      nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish())
        self.mean = nn.Linear(256, 32)

    def forward(self, x):
        return self.mean(self.features(x))


class ReleasedSuperMimic(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        p = torch.load(checkpoint, map_location="cpu", weights_only=False)["policy"]
        self.encoder = nn.ModuleList([nn.Linear(680, 2048), nn.Linear(2048, 1024),
                                      nn.Linear(1024, 512), nn.Linear(512, 256),
                                      nn.Linear(256, 32)])
        self.decoder = nn.ModuleList([nn.Linear(962, 2048), nn.Linear(2048, 1024),
                                      nn.Linear(1024, 512), nn.Linear(512, 256),
                                      nn.Linear(256, 29)])
        for i, layer in enumerate(self.encoder):
            layer.weight.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"encoders.motion.mlp.net.{2*i}.bias"])
        for i, layer in enumerate(self.decoder):
            layer.weight.data.copy_(p[f"decoder.mlp.net.{2*i}.weight"])
            layer.bias.data.copy_(p[f"decoder.mlp.net.{2*i}.bias"])
        self.register_buffer("goal_scale", p["encoders.motion.scale"].clone())
        self.register_buffer("obs_scale", p["decoder.scale"].clone())
        self.requires_grad_(False); self.eval()

    def encode(self, goal):
        x = goal * self.goal_scale
        for layer in self.encoder[:-1]:
            x = F.silu(layer(x))
        return torch.tanh(self.encoder[-1](x))

    def decode(self, obs, latent):
        x = torch.cat((latent, obs * self.obs_scale), dim=-1)
        for layer in self.decoder[:-1]:
            x = F.silu(layer(x))
        return self.decoder[-1](x)


class CurriculumLowLevel(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder = nn.ModuleList(list(policy.encoder))
        self.decoder = nn.ModuleList(list(policy.decoder))
        self.register_buffer("goal_scale", policy.goal_scale.clone())
        self.register_buffer("obs_scale", policy.obs_scale.clone())

    @torch.jit.export
    def encode_goal(self, goal):
        x = goal * self.goal_scale
        for i, layer in enumerate(self.encoder):
            if i < 4:
                x = F.silu(layer(x))
            else:
                return torch.tanh(layer(x))
        return x

    @torch.jit.export
    def decode_latent(self, obs, latent):
        x = torch.cat((latent, obs * self.obs_scale), dim=-1)
        for i, layer in enumerate(self.decoder):
            if i < 4:
                x = F.silu(layer(x))
            else:
                return layer(x)
        return x

    def forward(self, obs, goal):
        return self.decode_latent(obs, self.encode_goal(goal))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    from export_interactive_compliance import normalization

    for path in (CHECKPOINT, SM_CHECKPOINT, DEPLOY_CONFIG, FORCE_MANIFEST, TEMPLATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    task = state["cfg"].task
    if task.name != "G1SuperMimicAligned8OnlineLatentIncrementCurriculum":
        raise ValueError(f"Unexpected task: {task.name}")
    if task.action._target_ != "active_adaptation.envs.mdp.supermimic_residual.SuperMimicLatentIncrementCurriculumAction":
        raise ValueError(f"Unexpected action interface: {task.action._target_}")
    latent_scale = float(task.action.latent_increment_scale)
    if abs(latent_scale - 0.025) > 1e-8:
        raise ValueError(f"Unexpected latent increment scale: {latent_scale}")
    offsets = list(map(int, task.command.history_offsets))
    if offsets != [0, 1, 2, 4, 8, 16]:
        raise ValueError(f"Unexpected force history offsets: {offsets}")

    force_manifest = json.loads(FORCE_MANIFEST.read_text())
    if force_manifest.get("version") != "fixed8-aligned-observed-v1":
        raise ValueError(f"Unexpected force cache version: {force_manifest.get('version')}")
    force_points = []
    for i, point in enumerate(force_manifest["points"]):
        force_points.append({"name": point["body"], "body": point["body"],
                             "offset": list(point["offset"])})
    if len(force_points) != 8:
        raise ValueError(f"Expected eight force points; got {len(force_points)}")

    source = state["policy"]["actor"]
    obs_dim = int(source["module.0.module.1.module.0.weight"].shape[1])
    force_dim = len(offsets) * len(force_points) * 3
    base_dim = obs_dim - force_dim
    if (obs_dim, base_dim) != (452, 308):
        raise ValueError(f"Expected actor dims 308+144=452, got {base_dim}+{force_dim}={obs_dim}")
    actor = Actor(obs_dim).eval()
    feature_prefix = "module.0.module.1.module."
    mean_prefix = "module.0.module.2.module.actor_mean."
    actor.features.load_state_dict({k.removeprefix(feature_prefix): v for k, v in source.items()
                                    if k.startswith(feature_prefix)}, strict=True)
    actor.mean.load_state_dict({k.removeprefix(mean_prefix): v for k, v in source.items()
                                if k.startswith(mean_prefix)}, strict=True)

    deploy = json.loads(DEPLOY_CONFIG.read_text())
    sm_joints = list(deploy["joint_names"])
    template = json.loads(TEMPLATE.read_text())
    model_joints = list(template["joint_names"])
    if len(sm_joints) != 29 or set(sm_joints) != set(model_joints):
        raise ValueError("MuJoCo and SuperMimic joint names differ")
    model_to_sm = [sm_joints.index(n) for n in model_joints]
    low = ReleasedSuperMimic(SM_CHECKPOINT).eval()
    low_export = torch.jit.script(CurriculumLowLevel(low).eval())
    norm = normalization(state, obs_dim)
    if tuple(norm["loc"].shape) != (obs_dim,) or tuple(norm["scale"].shape) != (obs_dim,):
        raise ValueError("Unexpected actor normalization shape")
    # Force history is a separate TensorDict key, outside VecNorm. In this
    # task it remains raw root-local Newtons, with identity normalization.
    if not torch.equal(norm["loc"][-force_dim:], torch.zeros(force_dim)) or not torch.equal(
            norm["scale"][-force_dim:], torch.ones(force_dim)):
        raise ValueError("Aligned force history must remain raw N with identity normalization")

    if BUNDLE.exists() and any(BUNDLE.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty bundle: {BUNDLE}")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(BUNDLE / "student.pt"))
    torch.save(norm, BUNDLE / "normalization.pt")
    low_export.save(str(BUNDLE / "low_level.pt"))

    meta = copy.deepcopy(template)
    meta.update(
        backend="supermimic-aligned8-latent-increment-curriculum",
        checkpoint_name=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha(CHECKPOINT),
        supermimic_release="e11c_0220000", supermimic_checkpoint_sha256=sha(SM_CHECKPOINT),
        joint_names=model_joints, supermimic_joint_names=sm_joints,
        default_pos=[float(deploy["default_joint_pos"][i]) for i in model_to_sm],
        action_scale=[float(deploy["action_scale"][i]) for i in model_to_sm],
        supermimic_default_pos=[float(x) for x in deploy["default_joint_pos"]],
        supermimic_action_scale=[float(x) for x in deploy["action_scale"]],
        obs_dim=obs_dim, base_obs_dim=base_dim, force_history_offsets=offsets,
        force_history_slice=[base_dim, obs_dim], force_points=force_points,
        force_limit=30.0, force_slew=150.0, force_budget=60.0,
        all_body_force_points=False, latent_increment_scale=latent_scale,
        latent_dimension=32, latent_clip=[-1.0, 1.0], root_height=0.76,
        self_collisions=False,
        policy_role="SuperMimic aligned-8 latent-increment curriculum with contact-spring error reward",
        wandb_run="https://wandb.ai/luoxinyuan-duke-university/supermimic-aligned8-online-latent-increment-contact-spring-gpu23/runs/xzks5hv0",
        force_input="six samples at offsets [0,1,2,4,8,16] for ordered pelvis, left hand, right hand, torso point, left/right shoulder, left/right elbow; root-local raw Newtons",
        force_mapping={"cache_scale": float(task.command.force_scale),
                       "online_feedback_stiffness_N_per_m": float(task.command.online_feedback_stiffness),
                       "online_feedback_damping_Ns_per_m": float(task.command.online_feedback_damping),
                       "per_point_cap_N": 30.0, "total_cap_N": 60.0, "slew_N_per_s": 150.0,
                       "max_training_contacts": 3},
        latent_interface="At reset encode a stationary 10-frame goal from the measured starting pose. Every 50 Hz step accumulate 0.025*tanh(actor_output) in a persistent 32-D latent offset; decode clip(seed+offset,-1,1). No future GT is read during MuJoCo rollout.",
        interactive_goal="The training seed is the encoded initial GT 10-frame motion window. MuJoCo has no motion clip loaded, so it encodes ten repeats of the measured reset pose as the seed. This approximates the training initialization; no GT target is used after reset.",
        interactive_force_limitations="The interactive viewer applies a user-controlled spring force to one selected trained point at a time. Training replay may use up to three simultaneous cached contacts. The cached motion-conditioned feedforward component is unavailable without a GT clip; live drag forces still use the same eight point channels, root-local force-history convention, per-point/total caps and slew limit.",
        observation_terms=["root_ang_vel_history", "projected_gravity_history", "joint_pos_history",
                           "prev_actions", "sonic_joint_velocity", "latent_reference", "aligned_force_history"])

    torch.manual_seed(43)
    actor_probe = torch.randn(16, obs_dim)
    obs_probe = torch.randn(8, 930); goal_probe = torch.randn(8, 680); latent_probe = torch.randn(8, 32)
    with torch.inference_mode():
        actor_error = float((actor(actor_probe) - torch.jit.load(str(BUNDLE / "student.pt"))(actor_probe)).abs().max())
        enc_error = float((low.encode(goal_probe) - low_export.encode_goal(goal_probe)).abs().max())
        dec_error = float((low.decode(obs_probe, latent_probe) - low_export.decode_latent(obs_probe, latent_probe)).abs().max())
    if max(actor_error, enc_error, dec_error) > 1e-5:
        raise RuntimeError(f"Export parity mismatch: actor={actor_error}, encoder={enc_error}, decoder={dec_error}")
    meta.update(actor_export_max_error=actor_error, latent_encoder_export_max_error=enc_error,
                latent_decoder_export_max_error=dec_error, policy_dt=0.02)
    (BUNDLE / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (BUNDLE / "README.md").write_text(f"""# SuperMimic aligned 8KP contact-spring latent-increment policy

Exported from `{CHECKPOINT.relative_to(ROOT)}` (W&B run `xzks5hv0`).

## Run

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-aligned8-contact-spring-latent-increment
```

The actor consumes 308 state values followed by 144 raw root-local Newton values: eight ordered trained points sampled at `[0,1,2,4,8,16]` policy steps. Its output is a 32-D latent increment. Each 50 Hz update adds `0.025 * tanh(action)` to a persistent latent offset and decodes `clip(seed + offset, -1, 1)` using the frozen `e11c_0220000` SuperMimic decoder.

At reset the training policy seeds itself from the initial 10-frame GT motion window. No motion clip is loaded in this interactive demo; it encodes ten copies of the measured reset pose as an approximate seed. No future GT enters the rollout.

In the viewer, drag the pelvis, torso, shoulders, elbows, or hands to apply a spring force at the nearest trained point. Dragging controls one point at a time; the training force replay can use up to three concurrent contacts and also includes motion-conditioned cached feedforward forces. Those cached forces require a motion clip, so they are not synthesized here. Live forces use the trained point ordering, root-local Newton history, 30 N per-point cap, 60 N total cap, and 150 N/s slew limit.

The bundle includes the PPO actor, normalization, frozen low-level encoder/decoder, metadata and SHA256 manifest. Export checks compare the scripted actor and low-level components against their source implementations.
""")
    sums = {p.name: sha(p) for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"}
    (BUNDLE / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    print(json.dumps({"bundle": str(BUNDLE), "obs_dim": obs_dim, "base_dim": base_dim,
                      "force_dim": force_dim, "latent_increment_scale": latent_scale,
                      "actor_export_max_error": actor_error, "encoder_export_max_error": enc_error,
                      "decoder_export_max_error": dec_error}, indent=2))


if __name__ == "__main__":
    main()
