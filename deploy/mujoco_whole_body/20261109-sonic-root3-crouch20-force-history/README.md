# SONIC root3 force-history policy

Interactive MuJoCo bundle exported from the final checkpoint listed in
`metadata.json`. It uses the SONIC low-level G1 controller and the trained
29-joint reference residual actor.

The actor receives a 354-D normalized policy observation followed by six
root-local, 30 N-normalized force samples for pelvis, left hand EE and right
hand EE. Samples use control-step offsets `[0, 1, 2, 4, 8, 16]`. The force
history follows the training reset behavior and starts zero-filled.

Launch the browser demo with:

```bash
python scripts/mujoco_whole_body_web.py --policy sonic-root3-crouch20-force-history --port 8789
```

The official SONIC ONNX files are hash-checked and loaded from the shared
`.cache/mujoco-sonic/` cache; on first use they are downloaded from the pinned
official release.
