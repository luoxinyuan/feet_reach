# Multi10 segmented CEER student for MuJoCo

Exported from W&B run `4vqahv90`, final checkpoint `checkpoint_final.pt`. The checkpoint SHA-256 is recorded in `metadata.json`. This dated bundle coexists with the earlier `segmented/` and `segmented-static-hold/` policies.

Use with the standalone desktop or browser demo:

```bash
python scripts/mujoco_whole_body_student.py --policy segmented-multi10
python scripts/mujoco_whole_body_web.py --policy segmented-multi10 --port 8781
```

Record the single-robot frontal force sequence:

```bash
python scripts/mujoco_whole_body_force_demo.py --policy segmented-multi10 --output rollout.mp4
```

The MuJoCo runtime needs the exported student and frozen CEER low-level controller only; it does not need Isaac Lab, training data, motion clips, or the force cache.
