# CEER online-estimated root3 MuJoCo v2

This bundle uses the same trained residual actor, force estimator, and frozen
CEER low-level checkpoint as the original `20261030-ceer-online-estimated-root3`
bundle. The simulator adapter now restores the nested CEER action manager used
during Isaac Sim training: per-joint action scales, 0–4 physics-step delay,
low-level action smoothing/jitter, and boot protection.

Run the browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-estimated-root3 --port 8784
```

Run the desktop demo:

```bash
python scripts/mujoco_whole_body_student.py --policy ceer-online-estimated-root3
```

The alias `ceer-online-estimated-root3` selects this v2 bundle. The original
export remains available as `--policy ceer-online-estimated-root3-v1` for
comparison. Three input force points are pelvis, left EE, and right EE. The
actor receives only the bundled force estimator's predictions.
