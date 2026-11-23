"""Force-only encoder for the fixed 8KP SuperMimic curriculum."""
import torch
from torch import nn


FIXED8_FORCE_HISTORY_OFFSETS = (0, 1, 2, 4, 8, 16)


class Fixed8ForceEncoder(nn.Module):
    def __init__(self, in_features, hidden=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 32))
        self.register_buffer("force_mean", torch.zeros(in_features))
        self.register_buffer("force_scale", torch.ones(in_features))
        self.register_buffer("code_mean", torch.zeros(32))
        self.register_buffer("code_scale", torch.ones(32))

    def forward(self, force):
        latent = self.network((force - self.force_mean) / self.force_scale)
        return torch.where(force.abs().amax(-1, keepdim=True) > 1e-6, latent, 0.)

    def motion_code(self, latent):
        return latent * self.code_scale + self.code_mean


class Fixed8TemporalForceEncoder(nn.Module):
    """Compress the same causal 8 x 3 force channels at each history offset."""

    def __init__(self, steps, hidden=256, features="polar_delta"):
        super().__init__()
        self.steps = steps
        self.features = features
        width_per_point = {"polar": 7, "polar_delta": 10, "polar_delta_sum": 10,
                           "differences": 6, "signed_log": 6, "context": 9,
                           "pointwise": 3, "pointwise_delta": 6, "timewise": 3}[features]
        width = steps * (8 * width_per_point + (6 if features == "polar_delta_sum" else 0))
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_scale", torch.ones(width))
        self.contact_encoder = None
        if features.startswith("pointwise") or features == "timewise":
            local_input = 24 if features == "timewise" else steps * width_per_point
            self.contact_encoder = nn.Sequential(nn.Linear(local_input, 64), nn.SiLU(), nn.Linear(64, 16))
            blocks = steps if features == "timewise" else 8
            width = blocks * 16 + steps * 24
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 32))

    def lift(self, history):
        force = history.reshape(-1, self.steps, 8, 3)
        if self.features == "signed_log":
            return torch.cat((force, (force / .10).asinh()), dim=-1).flatten(1)
        if self.features in ("pointwise", "timewise"):
            return history
        if self.features in ("differences", "context", "pointwise_delta"):
            difference = torch.cat((torch.zeros_like(force[:, :1]), force[:, :-1] - force[:, 1:]), dim=1)
            features = [force, difference]
            if self.features == "context":
                net_other = force.sum(dim=-2, keepdim=True) - force
                features.append(net_other)
            return torch.cat(features, dim=-1).flatten(1)
        magnitude = force.norm(dim=-1, keepdim=True)
        direction = force / (magnitude + .05)
        features = [force, direction, torch.log1p(magnitude / .05)]
        if self.features != "polar":
            features.append((force[:, :1] - force).asinh())
        features = torch.cat(features, dim=-1).flatten(1)
        if self.features == "polar_delta_sum":
            net = force.sum(dim=-2)
            side = force[:, :, (1, 4, 6)].sum(dim=-2) - force[:, :, (2, 5, 7)].sum(dim=-2)
            features = torch.cat((features, net.flatten(1), side.flatten(1)), dim=-1)
        return features

    def encode(self, features):
        scaled = (features - self.feature_mean) / self.feature_scale
        if self.contact_encoder is not None:
            points = scaled.reshape(-1, self.steps, 8, scaled.shape[-1] // (8 * self.steps))
            local_input = points.flatten(2) if self.features == "timewise" else points.transpose(1, 2).flatten(2)
            local = self.contact_encoder(local_input).flatten(1)
            scaled = torch.cat((points[..., :3].reshape(len(points), -1), local), dim=1)
        return self.network(scaled)

    def forward(self, history):
        return torch.where(history.abs().amax(-1, keepdim=True) > 1e-6,
                           self.encode(self.lift(history)), 0.)


def load_fixed8_force_encoder(checkpoint):
    steps = len(checkpoint["history_offsets"])
    if checkpoint["version"] == "fixed8-force-encoder-v1":
        encoder = Fixed8ForceEncoder(24 * steps)
    elif checkpoint["version"] == "fixed8-force-encoder-v2":
        features = checkpoint["features"]
        hidden = checkpoint["hidden"]
        if features == "cartesian_latent":
            encoder = Fixed8ForceEncoder(24 * steps, hidden)
        else:
            encoder = Fixed8TemporalForceEncoder(steps, hidden, features)
    else:
        raise ValueError(f"Unsupported fixed8 force encoder version: {checkpoint['version']}")
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    return encoder.eval().requires_grad_(False)
