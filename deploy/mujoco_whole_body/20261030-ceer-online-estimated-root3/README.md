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

The policy and estimator TorchScript exports match their checkpoint networks exactly (maximum probe error 0). An 8-second headless idle playback fell at 3.8 s, and the estimator predicted nonzero force while no force was applied. The older direct-force `ceer-ee-root3` bundle also fell during an 8-second idle playback, so this smoke result appears shared with the current MuJoCo residual runtime/model and is not specific to the estimator export. The bundle itself loads and produces finite actions and estimates.
