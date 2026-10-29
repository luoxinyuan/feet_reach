# ceer-online-root3

Final residual student with frozen CEER. Input: robot state, self-generated joint reference, and actual root-local forces at pelvis/left hand/right hand. No motion dataset or external motion target is used at runtime.

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-root3 --port 8784
```

Start paused at standing; click Start then drag the robot. Per-point cap 30 N; total budget 60 N. Desktop viewer: `python scripts/mujoco_whole_body_student.py --policy ceer-online-root3`. Dependencies: see `scripts/references/README_mujoco_whole_body_student.md`.

Native actor export error: 8.94e-07. Low-level export error: 0. Native FK error: 1.33e-06.

30-second MuJoCo smoke test with 15 N hand pull: no fall, finite states. See validation.json. Web state and JPEG rendering checked; browser automation unavailable in this environment. This is a deployment check, not a robustness guarantee.
