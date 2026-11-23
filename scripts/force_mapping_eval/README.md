# Force-to-motion mapping evaluation v1

This is an offline **nominal-reference-state** audit, not policy evaluation.
It compares production 3-candidate and 9-candidate online Jacobian allocators
(both randomly activate 1–3 points) with cached 37-point minimum-force prompts.
All use the same frames and 60 N sum-of-point-norm / 30 N per-point caps.
The 37-point cache has K=2.5969108014710645; Jacobian methods retain K=80,
root-wrench objectives, regularization, selection, releases and slew rules.
This is an as-implemented comparison, not an isolated point-count ablation.

## Run

Use the gentle Python environment, from repository root:

```bash
python scripts/force_mapping_eval/prepare.py --output artifacts/force_mapping_eval/NEW_RUN
CUDA_VISIBLE_DEVICES=0 python scripts/force_mapping_eval/evaluate.py --input artifacts/force_mapping_eval/NEW_RUN
python scripts/force_mapping_eval/validate.py artifacts/force_mapping_eval/NEW_RUN
```

Preparation runs on CPU without Isaac. It executes the production allocator's
`_candidate`, `_select`, `_solve` method bodies extracted with Python AST, and
imports its production linear solver, direction projection and slew functions.
FK/J use exported robot geometry; joint columns are finite-difference tested.
Reference replay assumes actual q/root/velocities match the reference. Joint
feedback and velocity tracking error are zero. Root goals are 60*root position
increment and 8*yaw increment, with production caps. It is not a physical rollout
and does not validate a perturbed-state/online implementation.

The actor-visible force at t is the last physics interval's applied force.
The current desired solution is separately reported as `*_planned` to diagnose
smoothing/observation delay; it is not interchangeable with actor observations.
No force or state history enters the decoder.

## Protocol

Default 560 source clips, at most first 160 transitions/clip (50 Hz), ~80/20
original packed AMASS+EE+still and crouch corpus. Split 65/15/20 per source.
AMASS segments sharing a pre-segmentation source clip are grouped; only one is
sampled. Exact q-window duplicates are removed. Actor/subject identity is not
available here: this is not a subject-disjoint benchmark, and similarity between
crouch clips remains possible. All samples come from policy training data, held
out only from decoder fitting. Do not describe it as policy OOD evaluation.

Inputs: current 29 q, root-frame gravity, root height; force vectors in root
coordinates at a common fixed union of candidate locations, unoccupied points
zero. No velocity (which nearly reveals next Δq), targets, phase, or motion ID.
A matching state-only baseline uses zero force channels, identical network sizes.
Train-only input normalization; force std has a 0.1 N floor. Labels are next minus
current reference joint positions; scale 0.02 rad. Decoder 256/256 SiLU, AdamW,
100 epochs max, validation early stopping, 3 seeds. Checkpoints/scalers retained.

- Main error: sqrt(sum prediction squared error / sum target squared magnitude).
- Added force information: 1 - SSE(state+force)/SSE(state); clip bootstrap CI.
- Report arms, legs, waist, moving/static, source-specific errors.
- Shuffle force on test frames and inject 0.1 N component noise at active points.
- Conditional ambiguity: among 8 nearest cross-clip state neighbors, does choosing
  the closest force reduce target disparity? Report valid-neighbor coverage.
  This is descriptive; it is not a proof of injectivity.
- Force sum of norms, per-point peak, active count, root-coordinate slew.
- Analytical J^T F/K decoding of force37 is a capacity sanity check, not ranking.

A large gap between analytical and learned decoding means limited decoder/data
can dominate the score. Increasing force dimensionality may require more data;
poor MLP decoding alone does not establish non-invertibility. Nominal results
cannot establish robustness to actual-reference drift, contact forces, force
estimation, or human prompting. Future perturbed/rollout evaluation must use the
full online generator and an explicitly chosen feedback target.

## Root-motion extension and current comparison (v2)

The v2 extension keeps the same source-clip split and force inputs, and adds
root-local SE(3) labels to the decoder: next-frame and 5-frame relative root
translation plus the SO(3) log-map rotation, alongside joint Δq. Translation is
reported in meters, rotation and joint errors in radians. A shared 41-output
decoder predicts all targets; gains and NRMSE are computed separately for each
target group. At clip ends, the final four transitions are omitted so every
sample has a complete 5-frame root target.
This leaves 80,123 transitions from 560 source clips; train/validation/test
contain 363/84/113 clips and 52,541/11,706/15,876 transitions, respectively.

It adds the production 3KP direct spring: 80 N/m, pelvis and both hand points,
all three continuously active, 30 N/point and 60 N total caps, 150 N/s slew,
and zero force at reset. Force features use the actor-visible applied force,
including slew and observation delay. The comparison also includes the existing
3KP and 9KP Jacobian methods and 37-point offline force prompt.

### Held-out predictive results

Force gain is `1 - SSE(state+force)/SSE(state)`; positive values mean that force
improves prediction over the same state-only decoder. CIs bootstrap whole test
clips. NRMSE is lower-is-better. The v2 joint score is not directly comparable
to v1 because v2 fits a shared multi-output decoder and trims clip tails for the
5-frame root label.

|Method|Joint Δq NRMSE / gain|Root translation 1-frame NRMSE / gain|Root rotation 1-frame NRMSE / gain|Root translation 5-frame NRMSE / gain|Root rotation 5-frame NRMSE / gain|Mean force (N)|
|---|---:|---:|---:|---:|---:|---:|
|State only|0.905 / —|0.705 / —|0.841 / —|0.631 / —|0.801 / —|—|
|3KP Jacobian|0.820 / 17.9%|0.617 / 23.3%|0.788 / 12.1%|0.542 / 26.1%|0.752 / 11.8%|2.04|
|9KP Jacobian|0.900 / 1.2%|0.741 / -10.7%|0.884 / -10.6%|0.663 / -10.7%|0.854 / -13.8%|1.75|
|3KP direct spring|0.734 / 34.3%|0.398 / 68.1%|0.715 / 27.7%|0.313 / 75.4%|0.662 / 31.6%|1.23|
|37-point prompt|0.539 / 64.6%|0.591 / 29.6%|0.652 / 39.8%|0.541 / 26.4%|0.643 / 35.5%|5.22|

For 3KP Jacobian, spring, and 37-point forces, the clip-bootstrap 95% intervals
for each listed gain exclude zero. For 9KP Jacobian, every interval includes
zero. Full intervals and physical-unit RMSE/MAE are in the [v2 report](../../artifacts/force_mapping_eval/20261111_v2_root_valid/RESULTS.md).

### Interpretation

- **3KP direct spring is the strongest compact method in this nominal audit.**
  It improves prediction of joint changes and root translation/rotation at both
  horizons, while its test-set mean force is 1.23 N. Its root-translation gain
  is especially large. This is predictive correspondence in a reference-state
  replay, not evidence that a perturbed robot will physically move the same way.
- **3KP Jacobian adds consistent but smaller predictive value** across joint
  and root targets. **9KP Jacobian adds no detectable value with this decoder and
  split**, despite its slightly broader contact candidate set.
- **37-point force is best for joint Δq and first-frame root rotation**, and
  carries root-motion information. It uses about 4.2× the mean force of 3KP
  spring, can act on many more body points, and its offline cache is optimized
  over 64-frame blocks with future motion available. This is not a matched
  point-count or causal ablation.
- The spring/Jacobian comparison also changes allocation and contact schedule:
  spring keeps all three pelvis/hand anchors active; Jacobian randomly activates
  1–3. These scores identify promising mappings but do not isolate one algorithm
  variable.

### Reproduce v2

```bash
python scripts/force_mapping_eval/prepare_v2.py --output artifacts/force_mapping_eval/NEW_RUN
CUDA_VISIBLE_DEVICES=0 python scripts/force_mapping_eval/evaluate_v2.py --input artifacts/force_mapping_eval/NEW_RUN
python scripts/force_mapping_eval/validate_v2.py artifacts/force_mapping_eval/NEW_RUN
python scripts/force_mapping_eval/report_v2.py artifacts/force_mapping_eval/NEW_RUN
```

The completed run is `artifacts/force_mapping_eval/20261111_v2_root_valid/`.
Validation passed: source clips are split-disjoint, forces obey configured
caps, reset force is zero, and FK joint Jacobian finite-difference max error is
2.84e-7. The decoder was trained on GPU 0; no policy or Isaac rollout was run.

### Limits and next evidence

This remains an offline nominal-state decoder benchmark on the policy training
corpus, not a subject-disjoint or OOD test. It does not include actual-state
tracking error, contact dynamics, force estimator error, or closed-loop policy
response. The 37-point cache has future-trajectory access, and the compared
methods differ in mean force and active-contact schedule. The SSE reduction is a
decoder-dependent predictive score, not mutual information. For a final method
claim, repeat the correspondence score with the full force generator at
perturbed robot states, then run paired force-on/off rollouts using the same
initial states and motion references.
