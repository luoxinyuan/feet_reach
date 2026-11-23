# Aligned 8KP force latent for SuperMimic

The frozen encoder maps six causal snapshots of the eight applied, root-frame forces to a 32-dimensional vector. The actor receives this vector and robot observations and predicts a separate 32-dimensional SuperMimic control latent. Curriculum gradually removes reference assistance from the low-level decoder.

The shipped encoder has a mean NRMSE of 0.475 over three seeds on 113 held-out clips. The original raw-force readout scored 0.500 and the strongest widened raw readout scored 0.482. See [the offline benchmark](benchmark.json) for the five motion target groups and paired clip confidence intervals.

## Build the applied-force cache

Use the training environment with SuperMimic installed. Set `MEMPATH` to the directory containing `passive_whole_body_full/train` and `crouch_lowlevel/train`. Set `SUPER_MIMIC_ROOT` to the release directory containing `code` and `pretrained/e11c_0220000`. The attached solver, geometry and points reconstruct the cache for those datasets.

```bash
assets="$PWD/deploy/aligned8_force_latent"
export FORCE_CACHE="$PWD/artifacts/aligned8_force_cache"
python -m scripts.surface_contact_study.cache_aligned_prompts --init \
  --dataset "$MEMPATH" --output "$FORCE_CACHE" --workers 1 \
  --solver "$assets/solver.pt" --geometry "$assets/geometry.json" --points "$assets/points.json"
python -m scripts.surface_contact_study.cache_aligned_prompts --worker 0 \
  --dataset "$MEMPATH" --output "$FORCE_CACHE"
python -m scripts.surface_contact_study.cache_aligned_prompts --finalize --output "$FORCE_CACHE"
```

## Start RL

```bash
PYTHON_BIN="$(command -v python)" CUDA_VISIBLE_DEVICES=4,5 \
  FORCE_ENCODER_PATH="$assets/encoder.pt" \
  bash scripts/train_supermimic_aligned8_direct_latent_curriculum.sh
```

Omit `FORCE_ENCODER_PATH` to use the raw-force baseline. Set `ISAACLAB_ROOT` to your IsaacLab checkout and use `NUM_ENVS_PER_GPU` to fit local GPU memory. W&B logs online by default. Set `WANDB_MODE=offline` for local logging.

The encoder benchmark entry point is `python -m scripts.surface_contact_study.benchmark_fixed8_force_latent --help`. It takes the existing motion benchmark and aligned solver output files as inputs. Motion datasets and generated force caches are stored outside the repository.
