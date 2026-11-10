# Full-body force encoding baseline

This is an offline **joint-increment encoding** benchmark, not a dynamics simulation or a trained controller. Reference q[t] is the current pose; target is q[t+1]-q[t]. Floating-base translation/orientation changes are not target coordinates. No root-wrench matching objective is imposed.

## Force budget definition

`sum_i ||F_i||_2 <= B` at every frame. This is the sum of point-force magnitudes, **not** `||sum_i F_i||`, and not the coordinatewise L1 norm. Opposing forces consume budget independently. Each point also obeys `||F_i||_2 <= 30 N`. Smoothed variants obey `||F_i[t+1]-F_i[t]||_2 <= 150*0.02 N`. All bounds are inside the SOCP; outputs are not clipped afterward.

## Layouts and meaning

- `origins23`: the previous 23 non-mimic dataset link origins.
- `origins29`: origins of all 29 joint child links.
- `offsets37`: origins29 plus two non-collinear 5 cm local offsets on each hand mimic and each ankle-roll link (8 points).

Offset points are virtual body-attached diagnostic points; actual surface location, human accessibility and collision/contact feasibility have not been validated. The original USD geometry is read from `artifacts/ceer_segmented_jacobian/geometry.json`. All FK/Jacobians are world-expressed; only actuated joint columns enter the fit. Jacobians are verified against finite-difference FK, including distal wrist rotation.

`A=[J_1,q^T ... J_m,q^T]`, `tau=K*dq`, decoded increment `dq_hat=A*f/K`.

## Solvers

1. SVD pseudoinverse minimum **squared-force norm** baseline `f=A^dagger K*dq`; singular values below max(1e-10, 1e-7*s_max) are discarded. It has no force/slew constraints and is not a deployable force sequence.
2. Budget-only SOCP, two stages: minimize decoded-dq squared error subject to exact force budgets, then minimize mean summed point-force norms while allowing at most 1e-4 rad additional global RMS error above the first-stage optimum. A 1e-8 force-squared stabilizer resolves degeneracy.
3. Budget+smooth SOCP: same two-stage method with per-point temporal SOC bounds and additional `0.01/T * sum ||f[t+1]-f[t]||^2` in stage two. Forces are optimized jointly over the window. Start/end forces are free (not artificially zeroed), so deployment needs its own initial ramp.

The residual is scaled by 100 internally for numerical conditioning; reported errors are radians. Both stages must return Solved/AlmostSolved and pass explicit post-solve feasibility checks. Allowed numerical excess: total/per-point 1e-4 N, slew 1e-3 N/s; no hidden rescaling.

## Data and gain calibration

Sample up to three clips per available category/split from the fixed compliance-v1.1 manifest. Use 64 consecutive transitions near the center of each selected stored interval. This gives 27 train-diagnostic and 24 validation windows, 3,264 transitions total; results are a sampled baseline, **not a full-dataset precompute**. Categories are heuristic, and original AMASS recording-level split disjointness is unknown. This corpus includes crouch augmentation. The manifest records every dataset/clip/start frame and point layout.

First compare layouts/budgets at a fixed K=80, to avoid confusing gain changes with improved allocation. Then calibrate one global scalar K using only the 27 training windows, with the 37-point pinv solution:

`K = 80*min(20/p99(total_force), 30/p99(max_point_force), 150/p99(max_point_slew))`.

The second scan keeps this K fixed across every clip, frame and budget. Validation does not tune K. Calibration does not guarantee all frames meet budgets; the actual SOCP enforces them. Reducing K reduces required force but also increases decoding sensitivity to force measurement noise. A diagnostic independent isotropic component-noise sigma=0.1 N has expected decoded joint RMS `sigma*sqrt(mean_rows(sum_cols(A^2)))/K`; this is an assumed sensor-noise scenario, not a measured estimator error.

## Reproduce

Use the existing gentle Python environment. Install the isolated solver dependency (does not modify training packages):

```bash
python -m pip install --target artifacts/offline_force_baseline/tooling --no-deps clarabel==0.11.1
python scripts/offline_force_baseline/run.py --output artifacts/offline_force_baseline/EXPERIMENT_K80 --workers 8
python scripts/offline_force_baseline/calibrate.py --manifest artifacts/offline_force_baseline/EXPERIMENT_K80/manifest.json --output artifacts/offline_force_baseline/gain.json
# Set --gain to the scalar stored in gain.json:
python scripts/offline_force_baseline/run.py --output artifacts/offline_force_baseline/EXPERIMENT_SMALL_FORCE --layouts offsets37 --gain 2.5969108014710645 --budgets 5 10 20 --workers 4
```

Outputs: manifest; per-window JSON metrics/rank/joint errors/solver status; compressed NPZ with q, dq, FK points, singular values and every candidate force sequence; aggregate summary. Failed solves fail the run rather than silently dropping windows. Existing completed windows can be resumed only with the same manifest.

No policy training is started or altered by these scripts. No videos are produced; plots are standalone scientific artifacts. Before using the 37-point layout for human interaction training, validate actual contact points and force-estimator noise, then evaluate the resulting closed-loop policy separately.
