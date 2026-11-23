# SuperMimic 3kp Jacobian: genuinely reference-free low-level goal

The previous `SuperMimicResidualAction` consumed GT future motion plus residual.
This run uses a separate `ReferenceFreeSuperMimicResidualAction`; old checkpoint
behavior is preserved for reproducibility. The new residual is trained from
scratch, never resumed from those GT-conditioned runs. Surface-contact
experiments remain unlaunched.

## Information flow

Allowed: dataset -> online 3kp force generator -> physically applied force and
force-history actor input. Dataset also supplies rewards, critic privilege and
reference-state initialization. Initialization exposes only the actual initial
physical state to the actor/low-level, not its future.

Actor sees proprioception, its self-generated joint reference, and force history.
Its 29 actions update a persistent joint reference (2 rad/s, 0.6 rad reference
error bound). The frozen released SuperMimic receives **no dataset samples**:

- Ten 68D goal frames are synthesized from q_self + i*0.02*qd_self, i=0..9.
  Clip to joint limits; commanded velocities become zero at clipped limits.
- Joint positions are relative to SuperMimic's default pose, not reset origin.
- Upright gravity target [0,0,-1], roll/pitch angular-rate goals zero.
- Yaw rate, heading-frame linear velocity and root height use current measured
  physical state. No GT root orientation, height, velocity or heading is read.
- Ten frames are required by the frozen network's input shape. They are a causal
  constant-velocity preview of the residual's own command, not future GT frames.
- The 10-frame proprioception history consists only of current/past observations.

This changes the low-level goal distribution substantially. It is a real new
experiment; old good tracking results do not predict its outcome. Holding
measured root height/velocity is a causal interface choice, not an independently
learned root target. Upright root orientation remains a locomotion stabilization
prior; torso bending can be expressed through waist/limb references.

## Unchanged settings

3kp online Jacobian (pelvis and both hands), previous force generation and
physical limits (30 N/point, 60 N sum of norms), force history [0,1,2,4,8,16],
original mixed motion + 20% crouch data, dense tracking rewards with EE-specific
term disabled. Frozen released `e11c_0220000` checkpoint, 29D accumulated residual,
original DenseForcePPO teacher phase and PPO settings. Actor uses ground-truth
physical force history in this run, **not a trained force estimator**. Here
"teacher" means GT-force access; it does not mean GT-motion access for control.

## Validation

`scripts/reference_free_supermimic/smoke.py`:

- Replace command.dataset and command._motion with objects that throw on any
  access, then execute the actual low-level action callback; goal and action
  must exactly equal the normal run from the same reset state.
- Mutate critic-only GT observations and motion fields; public actor/observations
  must not change when physical state and supplied force are held fixed.
- 10 s zero-residual/no-force standing audit: root height >.55 m, tilt <.5 rad.
- Real Isaac PPO updates, finite losses, low-level frozen, exact DDP parameter
  synchronization, checkpoint round trip, force-budget assertions.

Results are under `artifacts/reference_free_supermimic`. These tests establish
input isolation and execution, not learned tracking quality.

## Launch

Use `/home/xl521/software/miniconda3/envs/gentle/bin/python` for tests, with
`PYTHONPATH=/home/xl521/IsaacLab/source/isaaclab:$PWD` and selected CUDA devices.

`bash scripts/train_supermimic_root3_reference_free.sh`

Defaults physical GPUs 5,6,7; 16,384 envs/GPU (49,152 total); 4B transitions;
W&B online project `supermimic-online-root3-crouch20-force-history`.
Launcher checks preflights and snapshots source, checkpoint hashes and reports.
`outputs/latest_supermimic_root3_reference_free.txt` records the launched output.
The old MuJoCo export uses a different standing-goal interface and must be
updated before deploying this new checkpoint; do not reuse it unmodified.
