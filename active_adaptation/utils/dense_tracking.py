"""Multi-scale per-component tracking rewards (units match the input error)."""
import torch


def component_tracking(error: torch.Tensor, sigmas):
    """Score each nonnegative component before averaging; keep [batch, 1]."""
    if not sigmas or any(s <= 0 for s in sigmas):
        raise ValueError('Tracking scales must be strictly positive')
    return torch.stack([torch.exp(-error.abs()/s) for s in sigmas]).mean(0).mean(-1,keepdim=True)
