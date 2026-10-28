# CEER EE-root 3-point compliance

Final checkpoint from the 2026-10-27 EE-root3 residual run. The force channels are pelvis, left EE, and right EE. MuJoCo dragging applies and observes the actual root-local force at each selected point.

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-ee-root3 --port 8782
```

This bundle contains the exported student, normalization, frozen CEER low-level controller, and policy metadata. No training data or Isaac installation is needed at runtime. SHA-256 values are in `SHA256SUMS.json`.
