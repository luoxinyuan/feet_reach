"""Reference-only dense translation prompts; no robot dynamics or target feedback."""
import numpy as np

EXCLUDED_POINTS = {'pelvis', 'head_mimic', 'left_hand_mimic', 'right_hand_mimic'}

def link_names(names, include_mimic=False):
    result = list(names) if include_mimic else [n for n in names if n not in EXCLUDED_POINTS]
    expected = 27 if include_mimic else 23
    if len(result) != expected:
        raise ValueError(f'Expected {expected} named points, found {len(result)}: {result}')
    return result

def displacement(positions, valid_next):
    delta = positions[1:] - positions[:-1]
    if not np.isfinite(delta).all():
        raise ValueError('Non-finite dataset positions')
    return np.where(valid_next[:, None, None], delta, 0.).astype(np.float32)

def cap_vectors(force, cap=30.):
    norm = np.linalg.norm(force, axis=-1, keepdims=True)
    return force * np.minimum(1., cap / np.maximum(norm, 1e-12))

def unit_metrics(delta, arms):
    norms = np.linalg.norm(delta, axis=-1)
    torque = np.cross(arms, delta).sum(axis=1)
    return dict(point_max=norms.max(axis=1), total=norms.sum(axis=1),
                net=np.linalg.norm(delta.sum(axis=1), axis=-1),
                moment=np.linalg.norm(torque, axis=-1), yaw=np.abs(torque[:, 2]))

def describe(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), p50=float(np.quantile(values,.5)),
                p95=float(np.quantile(values,.95)), p99=float(np.quantile(values,.99)),
                max=float(values.max()))

def choose_k(metrics, point_cap=30., typical_total=60., max_total=120., max_net=80.):
    # Fixed global gain. Limits are engineering screening budgets, not a proof
    # of dynamic stability. Train-only full-population statistics, no val tuning.
    constraints = dict(point_max=point_cap/max(metrics['point_max'].max(),1e-9),
        total_p99=typical_total/max(np.quantile(metrics['total'],.99),1e-9),
        total_max=max_total/max(metrics['total'].max(),1e-9),
        net_max=max_net/max(metrics['net'].max(),1e-9))
    return min(constraints.values()), constraints
