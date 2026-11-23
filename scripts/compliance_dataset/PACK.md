# ComplianceMix-29h-v1 training cache

The prepared dataset is `dataset/compliance_mix_29h_v1/`. It contains eight native `MotionData` memmap partitions under `train/<primary_behavior>`. There are 5,260,289 frames, 26,795 independent windows, 29.2238278 hours at50Hz. Storage is approximately8GB of float32/int32 arrays; original datasets and all selection manifests are unchanged.

## Use directly in current online compliance training

A complete example task is:

```text
task=G1/G1_supermimic_surface_observable5_compliance_mix
algo=ceer_force_prompt_teacher
```

It inherits the reference-free, frozen SuperMimic latent-residual max5 online-spring setup and replaces the data configuration. The reusable Hydra group `+compliance_data=compliance_mix_29h_v1` can replace data paths for other compatible online G1 tasks. Set `MEMPATH=/home/xl521/feet_reach/dataset` (or your dataset parent). Do not use old offline force caches with these new clip IDs. No force sequence is precomputed here.

Example single-process invocation, after choosing an available GPU:

```bash
MEMPATH="$PWD/dataset" \
PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD" \
ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 \
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/train.py \
 task=G1/G1_supermimic_surface_observable5_compliance_mix \
 algo=ceer_force_prompt_teacher checkpoint_path=null \
 task.num_envs=1024 wandb.mode=online
```

This is a usage example, not a launched full training run. Environment count should be profiled for the intended GPU setup. The cache stores33 bodies instead of the older27; full-scale buffer memory should be rechecked. Existing training buffer storage remains float16, source cache storage is float32.

## Sampling semantics

The eight path weights equal each category's selected duration divided by29.2238278h (6h single arms,4h both arms,5h locomotion,3h yaw,about2.225h crouch,3h bend,3h still,3h mixed). `resample_on_reset=true` exposes the entire corpus across training instead of permanently assigning a small fixed subset to environments. The production loader samples clips uniformly inside each category. These weights control category probabilities at resets; random start times, clip lengths and falls can change realized rollout-frame fractions. Do not claim exact frame balance in on-policy experience. Most selected clips are4s, minimum1s. No new train/validation split was invented because recording provenance is incomplete.

## Conversion

- Input poses already use standard G1 29-joint order, world Z-up, meters/radians, wxyz. No human-motion retargeting is repeated.
- Each approved window remains independent; no unapproved frame context, padding, duplicated holds or unrelated-motion concatenation is introduced.
- Normalize root quaternion lengths/sign continuity and recenter XY at the first frame. Full USD FK reconstructs all33 bodies, including wrist pitch/yaw and head/hand mimic links.
- Match the existing generator's ground convention: apply one constant Z translation so the minimum ankle-roll-link origin height in the window equals0. The runtime loader adds0.035m. This is the established ankle-origin convention, not a collision-based sole-height certificate. Applied XYZ translations are stored in provenance.
- Compute root linear velocity and joint velocity with centered within-window differences and one-sided endpoints at50Hz. Root angular velocity uses world-frame relative-rotation logarithms. No cross-window derivatives.
- Store float32 tensors and int32 motion_id/step. No additional angle clipping, speed filtering or dynamics filtering; the prior selected-motion content is preserved apart from coordinate normalization.

Each category folder contains `_tensordict/*.memmap`, `meta.json`, `meta_motion.json`, and `provenance.json` mapping each new motion ID to the dataset/source clip/original selected local span and subtype.

## Build and validate again (fresh output required)

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
 /home/xl521/software/miniconda3/envs/gentle/bin/python scripts/compliance_dataset/pack.py \
 --output /path/to/new/cache

PYTHONPATH="$PWD" /home/xl521/software/miniconda3/envs/gentle/bin/python \
 scripts/compliance_dataset/test_pack.py

MEMPATH="$PWD/dataset" PYTHONPATH="/home/xl521/IsaacLab/source/isaaclab:$PWD" \
 /home/xl521/software/miniconda3/envs/gentle/bin/python scripts/compliance_dataset/validate_pack.py
```

`manifest.json` records counts, transformations, hashes and conversion stats. `dataset_config.json` is the loader configuration. `validation.json` records full finite/boundary checks, independent NumPy FK, source-pose/velocity correspondence and production-loader sampling tests. `artifacts/compliance_dataset/compliance_mix_29h_v1/training_smoke/rank0.json` records an actual16-env physics/PPO integration test with frozen low-level, GT isolation and force cap/wrench checks. Passing a short smoke test establishes integration, not tracking quality or stability of all motions.
