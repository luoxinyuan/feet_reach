# SuperMimic 3KP Jacobian latent-increment curriculum policy

Exported from `outputs/20261117-supermimic-root3-jacobian-latent-increment-curriculum/161942-retry/train/wandb/run-20261117_161854-xkzi10ut/files/checkpoint_final.pt` (W&B run `xkzi10ut`). The actor has 449 inputs: 363 proprioceptive values, the current 32-D accumulated latent, and 54 values of root-local force history for pelvis and both hands. Its output is a 32-D latent increment.

## MuJoCo interaction

Run from the repository root:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-3kp-jacobian-latent-increment-curriculum
```

At reset, MuJoCo encodes ten repeats of the robot's measured starting pose and uses that as the seed latent. Every 50 Hz policy update adds `0.025 * tanh(action)` to one persistent latent offset, clips `seed + offset` to `[-1, 1]`, and decodes that latent using current SuperMimic proprioception. The offset is not restarted from the nominal pose each step. Future GT motion is not read at runtime.

The force prompt uses offsets `[0,1,2,4,8,16]`, three trained points (pelvis and both hands), root-local force components normalized by 30 N, and a 60 N total budget. In the viewer, left-drag the robot near the pelvis or either hand to apply force. The frozen low-level model is the pinned `e11c_0220000` SuperMimic release.

The training curriculum seeded each episode with the encoded initial GT motion window and gradually removed the future-GT latent base. The final stage used only the fixed episode seed plus accumulated policy increments. MuJoCo has no motion clip loaded, so its initial-pose seed is an interactive deployment approximation to that training seed. Metadata records the checkpoint hash and export parity errors.
