# CEER motion-tracking pipeline

This pipeline is an independent, force-free CEER low-level training path. It does not modify the existing CEER training configurations or scripts.

## Interface

- Actor policy observation: proprioception plus a 5D root-motion command (reference root height, planar velocity, heading).
- Teacher and student reference: the 145D full-body joint target is supplied directly in all three phases.
- No EE pose input, 3-point force interface, applied force input, force history, force estimator, or force-limit command.
- The root trajectory remains in the motion dataset and is used both for root tracking rewards and the actor's root-motion command.
- The existing privileged-state latent estimator remains part of the student adaptation workflow; it is unrelated to external force estimation.

## Rewards

The core tracking terms retain their existing weights: root position/rotation/linear velocity/angular velocity and joint position/velocity. The separate EE terms and compliance impedance rewards are absent. Survival, foot impact/slip, action-rate, joint-velocity, and joint-limit terms remain as stability regularizers. The fixed `feet_air_time_ref` prior is omitted so it cannot compete with the reference motion.

## Three phases

The existing `PPOPolicy` workflow remains unchanged: `ppo_train` learns the teacher, `ppo_adapt` trains the student adaptation module, and `ppo_finetune` updates the student policy. The joint-target estimator is bypassed because the full-body target is directly available to the student. Dataset mixture defaults to `passive_whole_body_full` plus `crouch_lowlevel` at weights 1.0 and 0.05, matching the recent crouch-capable low-level run.

## Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_ceer_motion_tracking_pipeline.sh
```

Frame budgets, environment count, Python executable, and output directory can be set with `TEACHER_FRAMES`, `ADAPT_FRAMES`, `STUDENT_FRAMES`, `NUM_ENVS`, `PYTHON_BIN`, and `OUTPUT_DIR`.
