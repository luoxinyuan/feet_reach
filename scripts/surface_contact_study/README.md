# Continuous surface contact force study

This is a force-prompt **teacher** experiment: force is applied in physics and
observed as ground truth. The actor sees proprioception, applied contact tokens,
and its own latent state. It does not see motion reference, phase, clip ID, future
joint targets, or planned but unapplied contacts. The critic/rewards/generator may
read motion labels. SuperMimic encoder and decoder are frozen.

## Force construction

Mesh samples are continuous barycentric points, not a fixed list of keypoints.
A temporary pool is drawn anew at each contact segment. For candidate point p,
its current geometric translational Jacobian is J_p. For a set S, let A_S stack
these Jacobians. With reference generalized displacement d_h at horizons
h = 0, 10, 20, 30 (20ms/frame), score:

    d_hat_h = A_S^T (A_S A_S^T + 0.002 I)^-1 A_S d_h
    E(S) = mean_h ||d_hat_h-d_h||²/(||d_h||²+1e-7) + 0.015 |S|
           + 0.30 * number_of_uncovered_preferred_regions

The three soft preferred regions are pelvis, left wrist/hand, and right wrist/hand
(pelvis and wrist_yaw links). This is an explicit human-interaction coverage prior,
not a learned guarantee of bijection. It does not fix contact coordinates or forbid
other upper-body links; the reconstruction improvement can outweigh the prior.
Candidates are continuously sampled on mesh triangles and lower-limb/floor
candidates are excluded. Head geometry belongs to torso_link in this mesh model.

Greedily add the candidate with lowest reconstruction error, then retain the best
prefix, at most 3 or at most 5 contacts. All joint increments are scored, including
stationary joints. Root displacement is world-frame metres; root rotation and
joint displacement are radians. This is a geometric proxy, not a torque/dynamics
model, nor proof of bijectivity or a globally optimal contact set.

Once selected, a contact stays attached to the same link/material surface point
for a 0.8s segment. Its next-frame anchor is obtained by full-body FK using the
same exported USD joint geometry as training:

    F_i_desired = 80 N/m * (x_ref_i[t+1] - x_actual_i[t])

Uniform scaling enforces max_i ||F_i|| <= 30N and sum_i ||F_i|| <= 60N.
No constraint drives the vector sum of forces to zero for stationary roots.
A common interpolation factor limits every contact's change to 150N/s at each
5ms physics substep; this preserves both convex force bounds. Unloading starts
when remaining segment time <= max_i ||F_i||/150 + 2*control_dt. Contacts change
only after every old force has decayed below 1e-5N; numerical residues are cleared.
Thus old and new sets do not overlap and violate the simultaneous-contact limit.
Invalid below-floor candidates are excluded. A zero-force static/fallen state is
allowed; “1–3/1–5” is a maximum active-contact budget, not compulsory nonzero force.
Surface accessibility and unilateral push/friction constraints are not modeled:
these are grasp/push-pull guiding contacts, not validated palm contacts.

## Evaluation protocol and iterations

Use the **valid** v2 benchmark (`20261111_v2_root_valid`), not the superseded v2
root-label bug run. Same 560 source-motion-grouped clips; 52,541 train / 11,706
validation / 15,876 test transitions. It is decoder-held-out data from a training
corpus, not policy tracking results. The pool was previously filtered for 3kp,
which favors the old 3kp family; there is no claim of universal coverage.

Targets: 29-D joint delta, root-local translation and rotation at 1 and 5 frames.
State input is q + projected gravity + root height, no velocity/phase/reference.
Matched 256/256 decoder, AdamW, validation early stopping. NRMSE is lower better;
compare against state-only, 3kp/9kp Jacobian, and the stronger direct spring3.
Use actor-visible previous-interval force, including slew/unloading delays.

Initial `pool64` and `pool128` experiments used per-link resultant force/moment
features and fixed 0.2s unloading. These failed to beat 3kp. They are retained as
failed development experiments, not selected results. Contact positions cannot
be fully recovered from a wrench, particularly parallel to the force; the next
iteration uses identical padded 5-contact tokens for **every** baseline: body
one-hot, body-local surface position, root-frame force, visibility. Inactive
contacts are exactly zero and tokens are sorted by physical identity/position.
All methods have the same decoder input width and capacity. `contact128` uses
adaptive unloading; `contact_upper128` additionally excludes leg surfaces.
Do not compare scores across the two feature protocols as if only the generator
changed. Compare methods within one matched protocol.

Further trials: global and state-conditioned training-only displacement covariance
priors did not improve validation enough and are **not used**. Full-segment scoring
plus a soft pelvis/left-hand/right-hand coverage cost did improve it. With three
seeds, increasing coverage cost .15 -> .30 changed joint NRMSE .7332 -> .7247
(max3) and .7669 -> .7655 (max5); gains were diminishing. The .30 version is frozen,
with covariance=None, upper_body=True, pool=128 and full_span=True. Its candidate
locations remain continuous; the source-region prior is intentionally explicit.

Final results are in `artifacts/surface_contact_study/coverage030_boundary_fixed/RESULTS.md`.
This final replay fixes a clip-end audit issue: lookahead frames clamp at the true
motion end, instead of repeating the final nonzero delta at the evaluation crop.
Parameters were frozen beforehand and were not retuned after observing test data.
The earlier coverage030 test is superseded only for that boundary correctness fix.

Development prints validation metrics only. Write `SELECTION.json` before using
`--final`; then evaluate the frozen selected generators with seeds 11/22/33 and
whole-clip bootstrap confidence intervals. Do not alter the selected method in
response to test scores.

Commands (from repository root, use the `gentle` Python environment):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/surface_contact_study/prepare.py \
  --pool 128 --upper-body --full-span --coverage-prior .3 \
  --output artifacts/surface_contact_study/coverage030_boundary_fixed
CUDA_VISIBLE_DEVICES=0 python scripts/surface_contact_study/evaluate.py \
  --input artifacts/surface_contact_study/coverage030_boundary_fixed
# Add --upper-body to prepare.py for the upper-body-only candidate ablation.
# After freezing SELECTION.json:
CUDA_VISIBLE_DEVICES=0 python scripts/surface_contact_study/evaluate.py \
  --input SELECTED_DIRECTORY --final --seeds 11 22 33
```

## Frozen latent controller

The released SuperMimic decoder accepts a single 32-D tanh-bounded latent plus
930-D proprioception history. `ReleasedSuperMimic.encode/decode` exactly factor
its original forward pass. Direct latent residual is therefore used, per the
requested preference; it is **not** a ten-latent-frame decoder input.

The anchor is a self-generated ten-frame hold of the reset pose, upright gravity,
and currently measured root velocities/height. It does not read future motion.
A moving hold-current-pose anchor failed a 10s stand test, because drift kept
redefining the reference. The reset-pose anchor passed. The residual endpoint
covers [-2,2]^32 so every decoder latent in [-1,1]^32 is reachable from the anchor:

    endpoint = 2*tanh(actor_action)
    residual += (endpoint-residual)/10
    z = clamp(encode(self_generated_hold) + residual, -1, 1)
    low_action = frozen_decode(proprio, z)

This is receding-horizon linear interpolation/first-order smoothing at 50Hz;
it does not guarantee arrival after ten steps if replanned every step. The
separate `EndpointInterpolationSuperMimicResidualAction` implements the literal
29-joint endpoint -> ten interpolated goal frames interface, but is not used in
these direct-latent experiments. Neither interface needs future GT.

## Validation and launch

- `test_math.py`: full-body FK vs independent NumPy USD FK, encode/decode equivalence.
- `standing.py --standing`: 10s zero residual, no guidance; forbids low-level GT access.
- `scripts/surface_compliance/smoke.py --mode observable3|observable5`: actual
  physics wrench checks, force budgets, contact permutation invariance, zero-force
  visibility, actor/low-level target isolation, PPO update, frozen low-level,
  checkpoint round-trip and multi-GPU parameter synchronization.
- `train_supermimic_observable_surface.sh`: snapshot code/geometry/checkpoint hashes,
  require standing and DDP checks, W&B online, 4B environment frames from scratch.
  MAX_CONTACTS=3 pins physical GPUs 2,3,4; MAX_CONTACTS=5 pins 5,6,7.

Offline mapping gains alone do not establish better compliance/tracking. Evaluate
trained policies separately with the standard dataset replay and fixed-force
behavior suite, including force removal, unilateral EE and yaw tasks.
