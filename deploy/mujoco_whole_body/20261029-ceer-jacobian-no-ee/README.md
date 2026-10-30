# CEER 9-point Jacobian residual, no EE tracking reward

Final checkpoint from the 2026-10-29 run `9td27f1q` in the
`ceer-jacobian-no-ee-residual-compliance` W&B project. It uses the online
9-candidate-point Jacobian force command. The separate EE tracking reward is
disabled; joint and keypoint tracking rewards remain enabled.

The policy receives current ground-truth applied forces at all nine candidate
points, expressed in the robot root frame. This bundle contains the TorchScript
student, its frozen observation normalization, the matching frozen CEER low-level
controller, and the metadata needed by the standalone MuJoCo demo. No Isaac Lab,
CUDA, motion data or force cache is needed at runtime.

Interactive browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy jacobian-no-ee --port 8785
```

Desktop demo:

```bash
python scripts/mujoco_whole_body_student.py --policy jacobian-no-ee
```

The MuJoCo run is a portable interaction demo, not a bit-exact reproduction of
Isaac Sim. The policy checkpoint and low-level controller hashes are recorded in
`metadata.json`; file hashes are in `SHA256SUMS.json`.
