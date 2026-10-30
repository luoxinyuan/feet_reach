# CEER online-estimated root3 MuJoCo bundle

Exported from the 2026-10-30 finetune checkpoint in `ceer-online-estimated-root3` (W&B run `4e69gvyj`). The actor receives 354 public proprioceptive/reference features and 9 estimated force values for pelvis, left EE, and right EE. The bundled estimator maps ten 93-dimensional proprioception/action frames to those three root-local force vectors. MuJoCo force ground truth is not fed to the actor.

Start the desktop viewer:

```bash
python scripts/mujoco_whole_body_student.py --policy ceer-online-estimated-root3
```

Start the browser viewer:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-estimated-root3 --port 8784
```

This bundle restores the frozen CEER low-level action configuration used in Isaac Sim: per-joint action scales, action delay, smoothing/jitter, and boot protection. The actor and estimator exports match their checkpoint networks exactly (maximum probe error 0). In the paired MuJoCo regression, this corrected bundle remained standing for 10 seconds with and without a 15 N left-hand pull; the previous export fell at 3.8 seconds even without an applied force.
