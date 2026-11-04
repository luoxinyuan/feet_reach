# 3-point CEER teacher with force history and crouch data

Exported from the final teacher checkpoint of W&B run `aej91ijk`. This is the
3-point pelvis/left-EE/right-EE policy trained with 20% crouch-data sampling and
the six force-history offsets `[0, 1, 2, 4, 8, 16]` control steps. The policy
receives normalized, root-local MuJoCo-applied force at each offset; the
observation begins with the 354-D proprioceptive vector and appends the 54-D
force-history vector. At reset, unavailable past force samples are zero-filled.

The frozen low-level controller is crouch-trained CEER checkpoint `r2e0ta3u`,
step 4500. This bundle includes the TorchScript high-level teacher, its
normalization, and the frozen low-level controller. It does not need the
training dataset, estimator, or Isaac Lab at runtime.

Browser demo:

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-online-root3-crouch20-force-history --port 8788
```

Desktop demo:

```bash
python scripts/mujoco_whole_body_student.py --policy ceer-online-root3-crouch20-force-history
```

Drag the pelvis or either hand to apply a force. Each force point is capped at
30 N; force arrows show the applied forces. Model provenance and artifact hashes
are recorded in `metadata.json` and `SHA256SUMS.json`.
