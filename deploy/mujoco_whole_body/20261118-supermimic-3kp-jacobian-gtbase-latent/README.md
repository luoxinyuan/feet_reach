# SuperMimic 3KP Jacobian GT-base-latent residual policy

Exported from `outputs/20261115-supermimic-root3-crouch20-force-history-latent/095800/train/wandb/run-20261115_095820-ebnq64fc/files/checkpoint_final.pt` (W&B run `ebnq64fc`). The actor consumes 363 proprioceptive values and six root-local force-history samples for pelvis and both hands (417 inputs total). The frozen SuperMimic encoder maps a 10-frame goal to one 32-D base latent. At each 50 Hz update the policy adds `0.1 * tanh(actor_output)` to that latent and passes it to the frozen decoder.

## MuJoCo interaction

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-3kp-jacobian-gtbase-latent
```

Left-drag the robot to apply forces at the nearest of the trained three Jacobian points: pelvis, left hand, or right hand. The tool uses the trained 30 N per-point cap, 60 N total budget, 150 N/s slew, and force-history offsets `[0,1,2,4,8,16]`.

## Reference conditioning

The Isaac benchmark for this checkpoint supplied the next ten ground-truth motion frames to the SuperMimic encoder on every step. This is reference-conditioned tracking, not a force-only motion-generation result; its reported benchmark MAE must not be interpreted as force-only performance. The interactive MuJoCo viewer has no motion dataset, so it uses a stationary nominal-pose goal to produce the base latent. It preserves the learned residual and force-history path, but does not reproduce the benchmark's future-GT input. No motion dataset or Isaac package is loaded at runtime.
