# SuperMimic 9-point Jacobian force-history policy

Standalone interactive MuJoCo bundle exported from the evaluated final teacher
checkpoint recorded in `metadata.json`. It includes the trained residual actor,
normalization, and a TorchScript copy of the pinned released SuperMimic G1
low-level tracker. The training checkpoint and original training data are not
needed at runtime.

The residual actor receives 354 proprioceptive/reference features followed by
six root-local force samples at control-step offsets `[0, 1, 2, 4, 8, 16]`.
Each sample has three force components for each point, ordered as torso, left
shoulder, left elbow, left wrist, left hand, right shoulder, right elbow,
right wrist, right hand. Force samples are divided by 30 N, and unavailable
history at reset is zero-filled as in training.

Launch the browser demo from the repository root:

```bash
python scripts/mujoco_whole_body_web.py --policy supermimic-9pt-crouch20-force-history --port 8792
```

Or use the desktop viewer:

```bash
python scripts/mujoco_whole_body_student.py --policy supermimic-9pt-crouch20-force-history
```

The interactive scene uses a stationary upright goal because it has no motion
clip. The trained reference residual is integrated and passed to the
SuperMimic low-level controller. `validation.json` records headless MuJoCo
smoke checks. These checks verify loading, finite state, standing, and force
history wiring; they are separate from the full IsaacLab compliance evaluation
at `artifacts/compliance_eval/20261110-supermimic-9pt-crouch20-force-history-v1_1/`.
The full benchmark reported 0.133 rad validation Joint MAE with 96.1% physical
completion.
