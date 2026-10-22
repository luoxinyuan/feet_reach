#!/usr/bin/env python3
"""Export the native CEER whole-body student, without Isaac at runtime.
Network reconstruction adapted from ceer_deploy/scripts/export_policy_vecnorm_from_checkpoint.py.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re, sys, types
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from torch import nn

class ActorStudent(nn.Module):
    def __init__(
        self,
        actor_input_dim: int,
        actor_hidden0: int,
        actor_hidden1: int,
        actor_hidden2: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.fc0 = nn.Linear(actor_input_dim, actor_hidden0)
        self.ln0 = nn.LayerNorm(actor_hidden0)
        self.fc1 = nn.Linear(actor_hidden0, actor_hidden1)
        self.ln1 = nn.LayerNorm(actor_hidden1)
        self.fc2 = nn.Linear(actor_hidden1, actor_hidden2)
        self.ln2 = nn.LayerNorm(actor_hidden2)
        self.mean = nn.Linear(actor_hidden2, action_dim)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln0(self.fc0(x)))
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.mean(x)

class StudentRollout(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        adapt_hidden0: int,
        adapt_hidden1: int,
        latent_dim: int,
        adapt_joint_hidden0: int,
        adapt_joint_hidden1: int,
        joint_target_dim: int,
        actor_hidden0: int,
        actor_hidden1: int,
        actor_hidden2: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Linear(obs_dim, adapt_hidden0),
            nn.LayerNorm(adapt_hidden0),
            nn.Mish(),
            nn.Linear(adapt_hidden0, adapt_hidden1),
            nn.LayerNorm(adapt_hidden1),
            nn.Mish(),
            nn.Linear(adapt_hidden1, latent_dim),
        )
        self.adapt_joint = nn.Sequential(
            nn.Linear(obs_dim, adapt_joint_hidden0),
            nn.LayerNorm(adapt_joint_hidden0),
            nn.Mish(),
            nn.Linear(adapt_joint_hidden0, adapt_joint_hidden1),
            nn.LayerNorm(adapt_joint_hidden1),
            nn.Mish(),
            nn.Linear(adapt_joint_hidden1, joint_target_dim),
        )
        self.actor = ActorStudent(
            actor_input_dim=obs_dim + latent_dim + joint_target_dim,
            actor_hidden0=actor_hidden0,
            actor_hidden1=actor_hidden1,
            actor_hidden2=actor_hidden2,
            action_dim=action_dim,
        )

    def forward(self, obs_key: torch.Tensor) -> torch.Tensor:
        latent = self.adapt(obs_key)
        joint_target = self.adapt_joint(obs_key)
        return self.actor(torch.cat([obs_key, latent, joint_target], dim=-1))

def ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = ensure_module(parent_name)
        setattr(parent, child_name, module)
    return module

def install_checkpoint_shims() -> None:
    ppo = ensure_module("active_adaptation.learning.ppo.ppo")

    class PPOConfig:
        pass

    class PPOPolicy:
        pass

    PPOConfig.__module__ = "active_adaptation.learning.ppo.ppo"
    PPOPolicy.__module__ = "active_adaptation.learning.ppo.ppo"
    ppo.PPOConfig = PPOConfig
    ppo.PPOPolicy = PPOPolicy

def load_checkpoint(path: Path) -> Mapping[str, Any]:
    install_checkpoint_shims()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}")
    return checkpoint

def require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    if key in mapping:
        value = mapping[key]
    else:
        value = mapping
        parts = key.split(".")
        idx = 0
        while idx < len(parts):
            if not isinstance(value, Mapping):
                raise KeyError(key)
            remaining = ".".join(parts[idx:])
            if remaining in value:
                value = value[remaining]
                idx = len(parts)
                break
            part = parts[idx]
            if part not in value:
                raise KeyError(key)
            value = value[part]
            idx += 1
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor at key {key}, got {type(value).__name__}")
    return value.detach().cpu()

def build_policy_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    policy = checkpoint["policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("checkpoint['policy'] is not a mapping")

    mapping = {
        "adapt.0.weight": "adapt_module.module.0.0.weight",
        "adapt.0.bias": "adapt_module.module.0.0.bias",
        "adapt.1.weight": "adapt_module.module.0.1.weight",
        "adapt.1.bias": "adapt_module.module.0.1.bias",
        "adapt.3.weight": "adapt_module.module.0.3.weight",
        "adapt.3.bias": "adapt_module.module.0.3.bias",
        "adapt.4.weight": "adapt_module.module.0.4.weight",
        "adapt.4.bias": "adapt_module.module.0.4.bias",
        "adapt.6.weight": "adapt_module.module.1.weight",
        "adapt.6.bias": "adapt_module.module.1.bias",
        "adapt_joint.0.weight": "adapt_joint_module.module.0.0.weight",
        "adapt_joint.0.bias": "adapt_joint_module.module.0.0.bias",
        "adapt_joint.1.weight": "adapt_joint_module.module.0.1.weight",
        "adapt_joint.1.bias": "adapt_joint_module.module.0.1.bias",
        "adapt_joint.3.weight": "adapt_joint_module.module.0.3.weight",
        "adapt_joint.3.bias": "adapt_joint_module.module.0.3.bias",
        "adapt_joint.4.weight": "adapt_joint_module.module.0.4.weight",
        "adapt_joint.4.bias": "adapt_joint_module.module.0.4.bias",
        "adapt_joint.6.weight": "adapt_joint_module.module.1.weight",
        "adapt_joint.6.bias": "adapt_joint_module.module.1.bias",
        "actor.fc0.weight": "actor_student.module.0.module.1.module.0.weight",
        "actor.fc0.bias": "actor_student.module.0.module.1.module.0.bias",
        "actor.ln0.weight": "actor_student.module.0.module.1.module.1.weight",
        "actor.ln0.bias": "actor_student.module.0.module.1.module.1.bias",
        "actor.fc1.weight": "actor_student.module.0.module.1.module.3.weight",
        "actor.fc1.bias": "actor_student.module.0.module.1.module.3.bias",
        "actor.ln1.weight": "actor_student.module.0.module.1.module.4.weight",
        "actor.ln1.bias": "actor_student.module.0.module.1.module.4.bias",
        "actor.fc2.weight": "actor_student.module.0.module.1.module.6.weight",
        "actor.fc2.bias": "actor_student.module.0.module.1.module.6.bias",
        "actor.ln2.weight": "actor_student.module.0.module.1.module.7.weight",
        "actor.ln2.bias": "actor_student.module.0.module.1.module.7.bias",
        "actor.mean.weight": "actor_student.module.0.module.2.module.actor_mean.weight",
        "actor.mean.bias": "actor_student.module.0.module.2.module.actor_mean.bias",
    }
    return {target: require_tensor(policy, source) for target, source in mapping.items()}

def build_student_rollout_from_state(state: Mapping[str, torch.Tensor]) -> StudentRollout:
    obs_dim = state["adapt.0.weight"].shape[1]
    adapt_hidden0 = state["adapt.0.weight"].shape[0]
    adapt_hidden1 = state["adapt.3.weight"].shape[0]
    latent_dim = state["adapt.6.weight"].shape[0]

    adapt_joint_hidden0 = state["adapt_joint.0.weight"].shape[0]
    adapt_joint_hidden1 = state["adapt_joint.3.weight"].shape[0]
    joint_target_dim = state["adapt_joint.6.weight"].shape[0]

    actor_input_dim = state["actor.fc0.weight"].shape[1]
    expected_actor_input_dim = obs_dim + latent_dim + joint_target_dim
    if actor_input_dim != expected_actor_input_dim:
        raise ValueError(
            "Unexpected actor input dimension: "
            f"actor.fc0 expects {actor_input_dim}, but obs+latent+joint is {expected_actor_input_dim}"
        )

    return StudentRollout(
        obs_dim=obs_dim,
        adapt_hidden0=adapt_hidden0,
        adapt_hidden1=adapt_hidden1,
        latent_dim=latent_dim,
        adapt_joint_hidden0=adapt_joint_hidden0,
        adapt_joint_hidden1=adapt_joint_hidden1,
        joint_target_dim=joint_target_dim,
        actor_hidden0=state["actor.fc0.weight"].shape[0],
        actor_hidden1=state["actor.fc1.weight"].shape[0],
        actor_hidden2=state["actor.fc2.weight"].shape[0],
        action_dim=state["actor.mean.weight"].shape[0],
    )

def export_policy(checkpoint: Mapping[str, Any], output_path: Path) -> StudentRollout:
    state = build_policy_state(checkpoint)
    model = build_student_rollout_from_state(state).eval()
    model.load_state_dict(state, strict=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output_path))
    return model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'outputs/passive-whole-body/20261021_231413/finetune/wandb/offline-run-20261022_033351-b0ak7xn3/files/checkpoint_final.pt'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/mujoco_whole_body_student')
    args = p.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    assert ckpt['policy']['last_phase'] == 'finetune', 'Expected a trained student'
    obs = ckpt['cfg']['task']['observation']['policy']
    assert list(k for k,v in obs.items() if v.get('enabled', True)) == ['root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','upper_body_forces_b']
    assert obs['joint_pos_history']['history_steps'] == [0,1,2,3,4]
    assert ckpt['env']['transforms.2.standard_normal'].item()
    args.output.mkdir(parents=True, exist_ok=True)
    net = export_policy(ckpt, args.output / 'student.pt')
    loc = ckpt['env']['transforms.2.loc'].cpu()
    scale = ckpt['env']['transforms.2.scale'].cpu()
    assert loc.shape == (265,) and (scale > 0).all()
    torch.save({'loc': loc, 'scale': scale}, args.output / 'normalization.pt')
    # Isaac uses breadth-first articulation order (also the existing CEER deployment order).
    joints = ['left_hip_pitch','right_hip_pitch','waist_yaw','left_hip_roll','right_hip_roll','waist_roll','left_hip_yaw','right_hip_yaw','waist_pitch','left_knee','right_knee','left_shoulder_pitch','right_shoulder_pitch','left_ankle_pitch','right_ankle_pitch','left_shoulder_roll','right_shoulder_roll','left_ankle_roll','right_ankle_roll','left_shoulder_yaw','right_shoulder_yaw','left_elbow','right_elbow','left_wrist_roll','right_wrist_roll','left_wrist_pitch','right_wrist_pitch','left_wrist_yaw','right_wrist_yaw']
    joints = [j+'_joint' for j in joints]
    # Read literal asset parameters without importing IsaacLab.
    asset = ast.parse((ROOT/'active_adaptation/assets/humanoid.py').read_text())
    params = {}
    for node in ast.walk(asset):
        if isinstance(node, ast.keyword) and node.arg in ('joint_pos','stiffness','damping','armature','effort_limit_sim') and isinstance(node.value, ast.Dict):
            params.setdefault(node.arg, ast.literal_eval(node.value))
    def expand(key):
        return [next((v for pat,v in params[key].items() if re.fullmatch(pat,j)),0.) for j in joints]
    meta = {'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'joint_names':joints, 'default_pos':expand('joint_pos'), 'kp':expand('stiffness'), 'kd':expand('damping'),
            'armature':expand('armature'), 'effort_limit':expand('effort_limit_sim'), 'policy_dt':0.02,
            'action_scale':0.5,'obs_dim':265, 'force_limit':30., 'force_slew':150.}
    with torch.inference_mode():
        torch.manual_seed(42)
        x = torch.randn(64,265)
        exported = torch.jit.load(str(args.output/'student.pt'))
        error = (net(x)-exported(x)).abs().max().item()
    assert error < 1e-6
    meta['torchscript_eager_max_error'] = error
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'obs_dim':265,'export_max_error':error},indent=2))

if __name__ == '__main__':
    main()
