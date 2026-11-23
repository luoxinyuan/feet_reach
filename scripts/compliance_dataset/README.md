# Common compliance dataset quality scan

Run from the feet_reach repository with NumPy (no GPU, Isaac, or training needed):

```bash
OPENBLAS_NUM_THREADS=1 /home/xl521/software/miniconda3/envs/gentle/bin/python \
  scripts/compliance_dataset/scan.py \
  --output artifacts/compliance_dataset/new_scan
python scripts/compliance_dataset/test_scan.py
```

Defaults scan the **unfiltered** BONES Seed G1 50Hz NPZ, LAFAN G1 50Hz NPZ, and the available AMASS G1 memmap cache. Override paths with `--boneseed`, `--lafan`, `--amass`; subset with `--datasets lafan amass`. NPZ qpos is loaded into RAM (BONES about 7.5GB), not qvel; cache arrays are read-only memory maps. AMASS FPS defaults to 50 because its metadata omits FPS; verify and override `--fps` for other caches. All inputs use world Z-up, root quaternion wxyz, meters/radians, standard MuJoCo G1 29-joint order. NPZ joint order follows the local converter's contract and cannot be inferred from the arrays alone.

## Selection

4s windows at 2s stride; include a final end-aligned window. Clips between 1s and 4s have one full-length window; clips below 1s are excluded. Overlap is intentional; union duration avoids double-counting. No train/test partition is generated: the AMASS cache lacks original recording IDs, so assigning random cache-clip splits would risk leakage. Resolve recording provenance before splitting.

Hard exclusions (defaults, adjustable by `--thresholds overrides.json`): nonfinite values; quaternion norm error >0.05; absolute joint angle >6.4rad (gross corruption gate, NOT hardware limits); adjacent-frame joint jump >0.6rad, root translation jump >0.25m, rotation jump >0.7rad; root height outside [0.15,1.5]m; root tilt >100 degrees (standing-interaction scope gate). Thresholds are initial engineering choices, not a validated definition of compliance suitability. Root height/tilt can exclude legitimate ground motions.

Fast motion is **only tagged**, not rejected: P95 maximum joint speed >8rad/s, root speed >2m/s, or maximum joint acceleration >150rad/s². Velocity uses a 100ms difference to reduce float16 quantization sensitivity. Acceleration remains a diagnostic, not a hard gate. Arm labels use maximum joint excursion >0.25rad; both-arm labels can overlap. Locomotion uses root net XY travel >0.25m; yaw range >0.3rad; crouch uses height range >0.1m plus knee range >0.3rad. Bend and still are heuristic labels. These are not human annotations or Cartesian EE coverage measures.

No collision/penetration checks, FK geometry checks, physical joint-limit validation, frozen-tracker rollouts, force-generator scores, weighting, or motion-data rewriting occur. Those are outside this first-pass quality scan.

## Outputs and consumption

- `<dataset>.allowlist.json`: source path/format, file sizes and mtimes, thresholds, accepted clip IDs and original NPZ names (AMASS has cache IDs only), accepted windows and merged intervals.
- `<dataset>.audit.jsonl`: one row per evaluated window, rejection reasons, labels, measurements.
- `summary.json`, `REPORT.md`: coverage, rejection counts, non-overlapping retained hours; script SHA256 in summary.

**Intervals are clip-local `[start,end)` frame ranges in the specified processed source at 50Hz. They are not original CSV/AMASS frames.** NPZ extraction would use `qpos[starts[clip_id]+start:starts[clip_id]+end]`; AMASS uses the same formula with its meta_motion starts and memmap arrays. The scan does not perform extraction. Keep window boundaries for sampling; merged intervals are an equivalent allowed-frame union, not a claim that a fresh 4s window across a union has been audited.

This versioned schema is deliberately distinct from `generate_dataset.py`'s legacy exact-segment allowlist. Do not pass it to that reader without an explicit adapter. Short clips, padding or preprocessing already baked into local caches cannot be reconstructed or undone. It assesses local processed data, not every upstream dataset release. Source sizes/mtimes are checked before/after; tests additionally check synthetic source bytes unchanged.
