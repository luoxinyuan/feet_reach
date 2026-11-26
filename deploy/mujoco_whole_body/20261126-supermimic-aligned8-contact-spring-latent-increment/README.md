# SuperMimic aligned 8KP contact-spring latent-increment policy

Exported from `outputs/20261125_194909-supermimic-aligned8-online-latent-increment-contact-spring-gpu23/train/wandb/run-20261125_195009-xzks5hv0/files/checkpoint_final.pt` (W&B run `xzks5hv0`).

## Run

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-aligned8-contact-spring-latent-increment
```

The actor consumes 308 state values followed by 144 raw root-local Newton values: eight ordered trained points sampled at `[0,1,2,4,8,16]` policy steps. Its output is a 32-D latent increment. Each 50 Hz update adds `0.025 * tanh(action)` to a persistent latent offset and decodes `clip(seed + offset, -1, 1)` using the frozen `e11c_0220000` SuperMimic decoder.

At reset the training policy seeds itself from the initial 10-frame GT motion window. No motion clip is loaded in this interactive demo; it encodes ten copies of the measured reset pose as an approximate seed. No future GT enters the rollout.

In the viewer, drag the pelvis, torso, shoulders, elbows, or hands to apply a spring force at the nearest trained point. Dragging controls one point at a time; the training force replay can use up to three concurrent contacts and also includes motion-conditioned cached feedforward forces. Those cached forces require a motion clip, so they are not synthesized here. Live forces use the trained point ordering, root-local Newton history, 30 N per-point cap, 60 N total cap, and 150 N/s slew limit.

The bundle includes the PPO actor, normalization, frozen low-level encoder/decoder, metadata and SHA256 manifest. Export checks compare the scripted actor and low-level components against their source implementations.
