# Gentle-contact A/B experiments

Both runs start from a fresh high-level policy with the same frozen CEER final
checkpoint, PointNet network, seed 0, reference dataset, wall distribution and
locomotion rewards. Each uses four GPUs and 4096 environments per GPU (16384 per
run). Teacher PPO uses 4 billion aggregate environment steps; adaptation uses
1 billion. There is no finetune stage. These are one-seed screening experiments,
not a multi-seed performance claim.

| Run | GPUs | Reward difference | tmux |
| --- | --- | --- | --- |
| A | 0–3 | Approach speed + task tracking/contact completion + impulse + residual smoothness | gentle-A |
| B | 4–7 | All of A, plus initial-contact force growth and peak cost | gentle-B |

## Shared task and reward

- Tracking weight 2, original `exp(-error/0.2)` kernel. In free space it uses full
  xyz error. If the nominal target lies >5 mm behind the wall and within its yz
  bounds, it uses tangential yz error. The gate depends on the original task,
  not whether the policy has touched the wall. The nominal command sent to CEER
  is not projected; the residual policy must learn to adjust it.
- Contact is allowed when the target is within 2 cm in front of the wall, or
  behind it, inside the wall's yz extent. More distant unexpected contact keeps
  the old -3 raw score. Required contact retains the old 160 ms grace, 160 ms
  ramp, and maximum -1.5 raw missed-contact score.
- Success: required contact in **every physics substep**, tangent error <=3 cm,
  continuously for 0.3 s. One actual +0.2 bonus per episode. Missing contact after
  success still incurs the normal miss cost.
- Approach: for a wall-relevant target, before contact, use positive world-x hand
  velocity (the wall normal points -x). Within 10 cm of the hand body origin,
  smoothly gate `min(((max(v-0.1,0)/0.2)^2),0.5)` with smoothstep of `1-distance/0.1`.
  Maximum cost after 20 ms dt is 0.01 per step. Geometry is synthetic PointNet
  wall data; privileged force/distance are used for reward only.
- Smoothness: `-0.01*min(sum((delta_residual/0.02)^2),5)` raw, where residual is
  the actual bounded xyz command in meters. Maximum actual cost 0.001 per step.
  No additional residual-amplitude penalty or full-body reference tracking.
- Impulse: accumulate the 200 Hz wall-filtered normal force over each initial
  100 ms contact window. Event cost is `min(6*(max(J-2,0)/1.1)^2,10)` raw, as in
  the previous impulse experiment. The maximum actual cost is 0.2 per event.

## B-only contact-window costs

- Positive force growth rate: `min((max(dF/dt-500,0)/5000)^2,0.5)`, averaged over
  physics substeps in each control step. Only the initial 100 ms window applies.
  Maximum actual integrated cost over a full window is 0.05.
- Peak: event cost `min(3*(max(Fpeak-40,0)/60)^2,5)` raw; maximum actual cost 0.1.
  Sustained force after the initial window has no added force-magnitude cost.
- No separate dense absolute-force term: this keeps the contrast limited to
  force growth and peak, rather than stacking another overlapping penalty.

All gentle penalties (approach, impulse, and B-only force penalties) are off for
the first 5% of teacher training, then linearly ramp to full scale at 20%.
Tracking, contact completion, locomotion and residual smoothing remain active.
Adaptation uses the full reward configuration and frozen observation normalization;
it trains the state estimator, not another PPO finetune stage.

## Event handling shared by A and B

A physical wall contact >1 N starts a window even if it occurs early. Another
window can start only after the previous window closes and force stays <0.5 N
for 100 ms. Thus a first soft touch does not exempt later collisions. Integrate
`F*physics_dt`, including the onset physics step: Isaac Lab obtains contact force
from PhysX with `dt=physics_dt`. This differs from the older post-detection
trapezoid definition; absolute J values should not be compared without matching
the integration convention.

Impulse and peak costs are paid as monotonic increments of their event costs,
instead of waiting until the window ends. The undiscounted total is the final
cost at a fixed curriculum factor. Early termination cannot erase accumulated
cost, though timing/discounting differs from an end-of-window bonus. No force
penalty continues outside the contact window.

The original command's `contact/mean_first_impulse_*` metrics are retained for
legacy diagnostics; they use the older first-expected-contact definition. Use
`gentle/*` and `tracking/gentle_*` for the new experiments. The gentle impulse
exceed fraction and P95 are current latest-event snapshots across environments
(including incomplete windows), not an unbiased completed-event distribution.

## Commands and outputs

```bash
GENTLE_VARIANT=A CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_gentle_contact_pipeline.sh
GENTLE_VARIANT=B CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_gentle_contact_pipeline.sh
```

W&B project: `wall-foot-reach`. Names:
`gentle-contact-A-v1-train`, `gentle-contact-A-v1-adapt`, and the corresponding B
names. Logs/checkpoints: `outputs/gentle-contact/{A,B}/<timestamp>`. Each variant
has its own `latest_pipeline.txt`; adapt starts only after a final teacher
checkpoint exists. `NUM_ENVS`, `TEACHER_FRAMES`, `ADAPT_FRAMES`, `LOW_CHECKPOINT`,
`PYTHON`, and `WANDB_RUN_NAME` can be overridden.

Validation: `tests/test_gentle_contact.py` covers event integration, cost caps,
first-step impact, repeated contact, short chatter, resets and no-wall behavior.
`scripts/references/smoke_proactive_residual.py --task G1_gentle_A` (or B) checks
physics/PPO, frozen CEER weights, task gates, speed penalty, one-shot success and
teacher-to-adapt loading. These are correctness checks, not trained-policy evals.
