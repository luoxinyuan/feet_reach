"""Observable surface-force command with an accumulated-latent curriculum."""
import torch

from .observable_surface import ObservableSurfaceCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.symmetry import SymmetryTransform


class ObservableSurfaceLatentCurriculumCommand(ObservableSurfaceCommand):
    """Keep arbitrary surface contacts while exposing the policy-owned latent."""

    def __init__(self, env, *, latent_curriculum_start=0.15,
                 latent_curriculum_end=0.80, **kwargs):
        super().__init__(env, **kwargs)
        self.latent_curriculum_start = float(latent_curriculum_start)
        self.latent_curriculum_end = float(latent_curriculum_end)
        if not 0.0 <= self.latent_curriculum_start < self.latent_curriculum_end <= 1.0:
            raise ValueError("latent curriculum bounds must satisfy 0 <= start < end <= 1")
        self.latent_curriculum_alpha = 0.0

    def step_schedule(self, progress: float, iter: int = 0):
        x = (float(progress) - self.latent_curriculum_start) / (
            self.latent_curriculum_end - self.latent_curriculum_start)
        x = min(max(x, 0.0), 1.0)
        self.latent_curriculum_alpha = x * x * (3.0 - 2.0 * x)
        self.env.extra["curriculum/latent_autonomous_fraction"] = self.latent_curriculum_alpha

    @observation
    def latent_reference(self):
        return self.env.action_manager.current_latent

    def latent_reference_sym(self):
        return SymmetryTransform(torch.arange(32), torch.ones(32))
