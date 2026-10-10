# Proactive EE tracking prototype

Task: double-support G1, left-hand marker XYZ in the **live root frame**.
The nominal references contain no walls and no braking conditioned on geometry.
An independent wall front face is sampled at x=0.40–0.50 m per episode;
25% of episodes have no wall (the collider is moved outside the workspace).
The target is not projected onto the wall. There is no distance-triggered mode
switch. The reward trades off EE tracking, balance, positive normal-force rise
and sustained normal force above a soft limit. Joint reference labels supervise
the target joint estimator, not an actor joint-tracking reward.

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

Default provisional reward thresholds: 20 N sustained normal force and
1000 N/s positive loading rate, with normalized squared hinge penalties.
They are experimental starting settings, not validated hardware limits.
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
