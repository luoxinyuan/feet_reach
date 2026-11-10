# SONIC 9-point Jacobian force-history policy

Interactive MuJoCo bundle exported from the completed SONIC teacher checkpoint
recorded in `metadata.json`. The low-level controller is the pinned SONIC G1
ONNX release, and the exported actor outputs the 29-joint reference residual.

The actor input is 354 proprioception features followed by six root-local force
samples at control-step offsets `[0, 1, 2, 4, 8, 16]`. Each sample contains
three force components for each point in this order: torso, left shoulder,
left elbow, left wrist, left hand, right shoulder, right elbow, right wrist,
right hand. Forces are divided by 30 N. History starts zero-filled, matching
the training reset behavior.

Launch the interactive browser demo from the repository root:

```bash
python scripts/mujoco_whole_body_web.py --policy sonic-9pt-crouch20-force-history --port 8791
```

The official SONIC ONNX files are hash-checked and loaded from the shared
`.cache/mujoco-sonic/` cache; on first use they are downloaded from the pinned
official release. The exported policy files and checksums are included here;
the full training checkpoint is referenced by path and SHA-256 in metadata.
