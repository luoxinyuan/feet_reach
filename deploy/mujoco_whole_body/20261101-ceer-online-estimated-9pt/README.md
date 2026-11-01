# CEER online-estimated 9-point compliance for MuJoCo

This bundle exports finetune checkpoint `qaxgsgg1` from the 2026-10-30 9-point
online-estimated-force training. The student estimates forces at the torso,
shoulders, elbows, wrists, and hands from ten frames of causal proprioception
and low-level action history. It does not receive force ground truth.

Run the desktop demo:

```bash
python scripts/mujoco_whole_body_student.py --policy ceer-online-estimated-9pt
```

Run the browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-estimated-9pt --port 8786
```

The mouse applies force at the nearest of the nine candidate link origins. The
force arrows show the applied force; the estimator HUD shows inferred force.
This policy inherits the prior 9-point experiment's disabled EE tracking reward.

The TorchScript actor and estimator match the training checkpoint exactly on
export probes. A 12-second headless MuJoCo test with a 15 N left-hand pull stayed
finite and above the fall threshold. The estimated total force averaged 12.8 N
before any force and 31.4 N during the pull, so MuJoCo force-estimation magnitude
is not yet calibrated reliably. This is a deployment smoke check, not an
equivalent Isaac Sim evaluation.
