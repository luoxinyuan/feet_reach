User deferred these experiments on 2026-11-13; do not launch them. Active replacement: `scripts/reference_free_supermimic/README.md`.

# Continuous-surface compliance variants

Status: both contact variants implemented; three-GPU preflights passed at
16,384 environments/GPU, two PPO updates each. Frozen weights, DDP consistency,
force/torque application and bounds passed. Final small test additionally checks
unapplied-contact masking and slot permutation invariance. Reports:
`artifacts/surface_compliance/smoke/summary.json`. Full training must wait for the
SuperMimic goal-interface choice. Existing `SuperMimicResidualAction` reads ten
future GT motion frames before adding the accumulated residual. It is reference
tracking with force-conditioned residual, **not force-only compliance**. The
launcher explicitly requires `SUPER_MIMIC_GOAL_MODE=reference_tracking`; do not
set this merely to bypass the unresolved design choice.

## Shared contact representation and physics

- True triangle/barycentric locations sampled from shipped G1 STL meshes; 20
  selected main links, 24 mesh parts including head attached to torso, hands,
  arms, pelvis, thighs, shins and feet. No fixed list of contact points.
- Link-uniform, then area-uniform mesh sampling. A floor-height rejection filter
  with bounded retries and torso fallback discourages contacts below 6 cm.
  This does not prove human reachability or lack of occlusion by other links.
- Grasp/attachment semantics permit signed forces. This is not a unilateral
  palm/contact-friction model.
- At most three active contact slots. Hold body-local locations for 1–3 s,
  ramp forces down before replacing the locations. Per-point <=30 N, sum of
  norms <=60 N, slew <=150 N/s, at 200 Hz physics / 50 Hz control.
- At actual robot pose, shift each COM Jacobian to the exact sampled surface
  position. Apply summed link force at link origin plus offset-induced moment;
  multiple points on one body are supported. No independent torque prompt.
- Joint prompt: 80*(q_ref[t+1]-q_ref[t])+20*(q_ref[t]-q_actual[t]). Regularized
  weighted least squares with previous-force smoothing, then budget scaling.
- Root translation/yaw terms activate for root motion or appreciable tracking
  drift. They are disabled for stationary, well-tracked root during local limb
  motion; therefore zero root intention does not impose zero total force.
- Actor: proprioception/self-generated reference plus observed contact body,
  local position, actual root-frame position, force and mask. Unforced contacts
  are hidden (zero token), including immediately after reset; selecting future
  contacts cannot itself leak a target to the actor before a force is applied. Slots are sorted
  canonically (permutation-invariant), history offsets [0,1,2,4,8,16]. No force
  estimator in this experiment. Critic receives dense reference privilege.

## Variants

`random`: continuous random positions and uniform contact count 1,2,3; same
force solver/limits as optimized.

`optimized`: sample 12 contact-set candidates, balanced across 1/2/3 contacts;
score weighted force-fit error over joint increments at t, t+4, t+8 using current
Jacobian/feedback (short-horizon local approximation), plus 0.025 per contact.
Select best set, then 4 projected-gradient steps on triangle barycentric
coordinates, accepting only objective improvements. Contact locations can change
triangles/links on subsequent resampling. Refinement stays on each selected
triangle. High residual for 10 steps after a 0.5 s minimum hold triggers early
release/reselection, enabling changes of motion mode. It is finite population
search plus local optimization, not a globally optimal mixed-integer solution.

## Files and commands

Use `/home/xl521/software/miniconda3/envs/gentle/bin/python`.

1. `scripts/surface_compliance/build_surface_mesh.py` exports mesh triangles.
2. `scripts/surface_compliance/test_geometry.py` checks Jacobian point shift,
   virtual work, objective gradients, simplex and force/slew constraints.
3. `scripts/surface_compliance/smoke.py --mode random|optimized --num-envs N
   --output PATH` runs Isaac and PPO, verifies actual force/torque equivalence,
   actor target isolation, low-level frozen weights and optional DDP sync.
   Set PYTHONPATH=/home/xl521/IsaacLab/source/isaaclab:$PWD and CUDA_VISIBLE_DEVICES.
4. `scripts/train_supermimic_surface.sh` snapshots source/meshes, logs checkpoint
   hashes, and launches online W&B project `supermimic-surface-compliance`.
   CONTACT_MODE=random defaults physical GPUs 5,6,7; optimized defaults 2,3,4.
   NUM_ENVS_PER_GPU and TOTAL_FRAMES are configurable (defaults 16384 and 4B).
   Do not launch until the goal-interface question is resolved.

Both use AMASS/EE/still + crouch data, weights .8/.2, the existing dense tracking
rewards (EE-specific term disabled), frozen released SuperMimic, and train only
29-joint accumulated residual PPO. No existing runs are modified.
