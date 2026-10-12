# Proactive EE tracking prototype

Task: double-support G1, left-hand marker XYZ in the **live root frame**.
The nominal references contain no walls and no braking conditioned on geometry.
An independent wall front face is sampled at x=0.40–0.50 m per episode;
25% of episodes have no wall (the collider is moved outside the workspace).
The target is not projected onto the wall. There is no distance-triggered mode
switch. The reward trades off EE tracking, balance and positive normal-force
rise above 500 N/s. There is no hand contact-force magnitude penalty.
The reward retains the original `G1_3kp` tracking and locomotion terms, weights
and tracking scales. Joint position/velocity tracking covers all 29 joints;
references also supervise the student whole-body joint estimator.
The EE term is `left_ee_tracking` (2.0), using the original low-level
`exp(-distance / 0.2)` kernel for the commanded left hand only. Its target
exactly matches the root-frame xyz exposed to the student, including external
commands. The extra left-hand reach (5.0) and double-support (3.0) terms have
been removed. The right arm retains full-body joint-reference constraints.
Legacy impedance terms that
require virtual force targets are not included. The standard finetune overrides
for joint velocity L2 (0.002) and action rate (0.5) remain unchanged.

The previously trained proactive v1 checkpoints omitted joint tracking rewards;
changing this config does not repair those checkpoints. Their evaluation
reports describe the old objective. Training remains stopped pending a new run.

## Data

```bash
OPENBLAS_NUM_THREADS=1 .venv-reference/bin/python scripts/references/generate_ee_reach_dataset.py
python scripts/data_process/pack_foot_reach_dataset.py --source artifacts/ee_reach --output dataset/ee_reach
```

Generation refuses to replace an existing manifest. Uses the original G1 USD
inertial model, matching the earlier foot-reach workflow. Crocoddyl optimizes
seven arm joints at endpoints with a fixed hand orientation. Root, both legs,
waist and right arm stay fixed; quintic trajectories are checked at 50 Hz for
joint limits, fixed feet, selected arm/body mesh pairs and double-support
inverse dynamics (friction pyramids, foot moments, 15% effort reserve).
The selected collision pairs are recorded in the manifest; this is not an
exhaustive swept-volume or real-hardware safety certificate.
Split is by endpoint, with both speeds in the same split. Metadata and extra
inverse-dynamics arrays stay in NPZ; the existing memmap loader packs training
kinematics in the same 29-joint / 27-body format as foot reach.

## Geometry and policy

128 metric xyz points plus a validity mask, in the live root frame. The cloud
is synthetic surface sampling of the wall front face, with 3 mm Gaussian noise,
10% point dropout and a 25 Hz update rate. Cached world points are transformed
to the current root at each control tick. It is **not rendered depth**, does not
model robot occlusion, and has no actual SLAM input yet. An empty valid set means
no observed surface, not proof that space is free. Real depth/SLAM integration
must preserve the same coordinate convention and test missing/old geometry.

The `geometry_` observation bypasses VecNorm to retain metric scale and masks.
A shared per-point MLP 3→32→64→64, masked max pooling and 64-D projection feeds
both teacher and student actors directly. Critic has its own PointNet. The joint
estimator remains a nominal EE/proprioception predictor. Teacher→student actor
copy includes PointNet. No teacher labels or privileged state are required by
the deployed student actor.

The positive loading-rate threshold is 500 N/s; its weight remains 0.1.
For each physics sample, `F = max(0, -left_hand_net_force_world_x)` and
`rate = max(0, (F - previous_F) / 0.005)`. Within each 20 ms control interval,
keep the maximum rate. The weighted cost is
`-0.1 * max(0, (rate_peak - 500) / 500)^2`.
The same threshold also supplies the normalization scale, preserving the
previous normalized-hinge convention. There is no force-magnitude reward cost;
force magnitude remains logged for evaluation. The legacy `contact_force_cost`
method remains available only for loading old checkpoint configurations.
This uses the left-hand **net** contact sensor, not a hand/wall pair filter:
other contacts can contribute; tangential force and unloading are not penalized.
There is no penalty clipping and no explicit pre-contact speed reward.
The standard environment time scaling multiplies reward by the 0.02 s control
period. A large steady contact force has zero loading-rate penalty.
Force peaks and loading-rate peaks are captured at every 200 Hz physics step;
reward uses maxima across the four substeps of the 50 Hz control tick.

## Verify and train

Use the project's IsaacLab PYTHONPATH, gentle Python and MEMPATH=dataset.

```bash
python scripts/references/smoke_proactive_ee.py --phase train
python scripts/references/smoke_proactive_ee.py --phase adapt --checkpoint artifacts/ee_reach/smoke_train.pt
python scripts/references/smoke_proactive_ee.py --phase finetune --checkpoint artifacts/ee_reach/smoke_adapt.pt
bash scripts/train_proactive_ee_pipeline.sh
```

Eight GPUs, 4096 environments each (32768 total); teacher/adapt/finetune budgets
4B/1B/2B aggregate frames. W&B project defaults to wall-foot-reach; runs use
proactive-ee-v1, proactive-ee-v1-adapt, proactive-ee-v1-finetune. Results are
isolated in outputs/proactive-ee. Override NUM_ENVS, WANDB_PROJECT or
WANDB_RUN_NAME through environment variables.

Passing integration checks only establishes working data, observations,
contact feedback and PPO updates. Proactive compliance still requires matched
blind-vs-geometry evaluation with pre-contact velocity, peak force, force slew,
steady force, completion time and free-space tracking metrics.

## Student evaluation with recordings

```bash
bash scripts/eval_proactive_ee.sh --checkpoint /path/to/adapt/checkpoint_final.pt \
  --video --masked-control --output artifacts/proactive_ee_eval/my_experiment
```

Default: seven seeded random held-out endpoints ordered by nearest neighbor,
continuous free-space reaching (2 seconds reaching, 1 second hold per target).
Then the nearest held-out target at least 1 cm behind x=0.44 m is tested with and without the wall,
three matched seeds per condition (3 seconds reach, 2 seconds hold). The paired
experiment starts directly from the identical reference state and observation
history; the wall is fixed from reset. `--paired-settle` optionally adds a
masked-geometry settling phase, but its resulting state differences must be
checked before drawing causal conclusions. The optional masked control retains the physical wall but masks the
PointNet input: this is an OOD diagnostic, not a separately trained blind policy.
No automatic resets occur within trials, and instability stops that trial.
Student receives only normalized proprioception/EE target and metric geometry.

The experiment folder contains the checkpoint hash, targets/protocol, 200 Hz
samples, JSON summaries, plots and per-condition MP4 recordings. Red is the
command target, green is the actual hand marker; screen rings remain visible
when occluded by the wall. Contact force is filtered to the hand-wall pair.
Metrics include hold error/RMSE/P95, success within 3 cm for >=95% of a completed
hold, time to reach, pre-detected-contact approach speed, peak normal force,
positive loading rate, impulse, steady force, stability and completion.
Compare precontact velocities in matched time windows, not just post-contact
slowing. A wall/no-wall contrast alone cannot establish learned compliance.

Recompute report tables without launching Isaac:

```bash
python scripts/analyze_proactive_ee.py artifacts/proactive_ee_eval/my_experiment
```

## Full-reference teacher / student arm-motion diagnostic

```bash
bash scripts/eval_proactive_reference.sh --checkpoint /path/to/train/checkpoint_final.pt \
  --output artifacts/proactive_ee_eval/teacher_reference --clips 0 4 8
bash scripts/eval_proactive_reference.sh --checkpoint /path/to/adapt/checkpoint_final.pt \
  --output artifacts/proactive_ee_eval/student_reference --clips 0 4 8
```

These runs follow complete held-out reference clips with no wall and no external
step-target override. The checkpoint phase selects the teacher (direct 29-joint
reference plus privileged observations) or student (EE xyz/proprioception plus
empty geometry). Both use deterministic actions and seed 42. Recordings are
25 FPS; NPZ logs contain actual/reference joint positions, measured velocities
and EE/feet/root errors at 50 Hz. Right-arm references are constant, making
unnecessary right-arm motion directly measurable. Velocity RMS includes all
motion, not just a frequency-filtered definition of jitter. A stability threshold
crossing is reported but the diagnostic continues unless root drift exceeds
0.5 m or joint positions become nonfinite. This differs from the earlier
step-target evaluation, which aborts on the first stability failure.

## Gradual wall insertion evaluation

```bash
bash scripts/eval_proactive_ee.sh --checkpoint /path/to/finetune/checkpoint_final.pt \
  --paired-trajectory --wall-x 0.43 --masked-control --video --repeats 3 \
  --output artifacts/proactive_ee_eval/gradual_wall
```

The paired protocol has a fixed **world** target path, converted to live-root
xyz at each control tick: 2 s quintic approach to 6 cm before the wall at the
endpoint height/y, 6 s quintic insertion to a held-out endpoint inside the
5 cm thick wall, and 4 s hold. The same schedule is used with no wall, with
wall/geometry, and with wall/geometry masked. No settling precedes the paired
trials, preserving identical initial robot state. Masking is an OOD diagnostic,
not a separately trained blind baseline. Free-space endpoint accuracy remains
the original step-command protocol and is reported separately.

Videos use a side-oblique camera, through-wall red target/green hand rings,
and an x-z section inset. The green dot is the hand body marker, not an exact
collision manifold point. Video is 25 FPS and force logs are 200 Hz.
`planned_trajectory.csv` saves the world-space schedule.

`--continue-on-instability` is a separate diagnostic mode: retain every original
stability violation in the results, but continue recording until root drift
exceeds 0.5 m, root height drops below 0.35 m, or state becomes nonfinite.
`trajectory_finished` means the full requested schedule ran; `completed` and
`tracking_success` remain false after a stability failure. Do not present this
mode as passing the original fixed-support evaluation. Analyze saved results
with `python scripts/analyze_proactive_ee.py EXPERIMENT_DIRECTORY`.
