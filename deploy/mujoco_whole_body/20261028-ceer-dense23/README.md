# CEER dense 23-point compliance

Final dense-force finetune checkpoint from the 2026-10-28 run. The 23 force points are the torso and limb link origins listed in `metadata.json`.

The policy does not receive applied-force ground truth. `estimator.pt` predicts the 23 root-local forces from ten recent proprioception and low-level-action frames; its output is passed to `student.pt`. The interactive viewer displays the estimator's summed magnitude, orange arrows for the actual forces, and a blue arrow plus magnitude for the selected point's estimate.

```bash
python scripts/mujoco_whole_body_web.py --policy ceer-dense23 --port 8783
```

This bundle contains the student, force estimator, normalization, frozen CEER low-level controller, and metadata. No training data or Isaac installation is needed at runtime. SHA-256 values are in `SHA256SUMS.json`.
