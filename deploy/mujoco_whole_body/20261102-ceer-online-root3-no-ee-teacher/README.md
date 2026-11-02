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

The MuJoCo adapter concatenates the 354 proprioception values followed by the
9 applied-force values, matching the teacher's training input order. A previous
adapter placed force between proprioception terms and caused the robot to fall;
that observation ordering is fixed. Headless checks now remain upright for 10 s
without force and through a scripted 15 N hand pull. TorchScript actor and
low-level exports match their checkpoint networks (max probe error 0);
provenance and file hashes are in `metadata.json` and `SHA256SUMS.json`.
