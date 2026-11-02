# 3-point no-EE-reward GT-force teacher

Exported from the completed teacher phase of W&B run `y3x1ilap`. The actor uses
MuJoCo's applied force at pelvis, left hand, and right hand in root-local
coordinates. It does not use the later student force estimator. Its frozen
low-level controller is the crouch-trained CEER checkpoint `r2e0ta3u`, step
4500.

Run the browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-root3-no-ee-teacher --port 8787
```

Run the desktop viewer with
`python scripts/mujoco_whole_body_student.py --policy ceer-online-root3-no-ee-teacher`.
The three force channels match `pelvis`, `left_hand_mimic`, and
`right_hand_mimic`; force arrows show the applied load.

This export is functional but its initial MuJoCo no-force smoke rollout fell at
1.33 s. The policy was trained in Isaac Sim, so this is a sim-to-sim limitation,
not evidence that its Isaac training rollout fell. Compare behavior cautiously.
TorchScript actor and low-level exports match their checkpoint networks (max
probe error 0); provenance and file hashes are in `metadata.json` and
`SHA256SUMS.json`.
