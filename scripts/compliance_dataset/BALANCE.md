# Behavior-balanced training allowlist (<=30 hours)

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
 /home/xl521/software/miniconda3/envs/gentle/bin/python \
 scripts/compliance_dataset/balance.py \
 --input artifacts/compliance_dataset/20261115_stage2_sparse_v1 \
 --max-contacts 5 --budget-hours 30 \
 --output artifacts/compliance_dataset/balanced_30h

/home/xl521/software/miniconda3/envs/gentle/bin/python \
 scripts/compliance_dataset/test_balance.py
```

Run from the repository root. Requires NumPy, PyTorch and existing USD-exported geometry/mesh metadata. GPU 0 was idle for the first run; training GPUs 2–7 were not used. Input motion files are read-only. No dataset arrays are rewritten, copied or resampled. A fresh output directory is required.

## Label extraction

Take 17 uniform samples from each stage-2 accepted window and reconstruct wrist origins and torso axis with G1 FK. Hand motion is in the full instantaneous root frame, excluding whole-body translation/rotation. Root translation and torso lean directions use heading-only coordinates (forward/left/up). Temporal displacement contributions, not merely final displacement, determine the dominant direction, so a reach-and-return has a nonzero motion label.

Motion activation thresholds: hand Cartesian range >=0.08m AND arm joint excursion >=0.2rad; root path >=0.25m; root yaw range >=0.3rad; root height excursion >=0.08m AND knee excursion >=0.25rad for crouch; torso lean excursion >=0.2rad for bend. Primary family salience scales are hand0.25m, root path0.5m, yaw0.7rad, height0.18m, lean0.35rad. If the second active family's salience reaches 65% of the first, assign mixed. Otherwise use the dominant family. Single/both-arm primary labels depend on active arms. Only stage-1 still-labelled windows become still; subthreshold non-still motion is unclassified and not selected. These are deterministic heuristics, not human annotations.

Subtypes: single-arm left/right × six directions; both-arm same/opposite/independent velocity coordination; root four directions × start/sustained/stop; left/right yaw; lowering/rising; forward/back/left/right bend; held low/lean/raised-hand/upright-other postures; mixed-family pairs. Root start/stop uses mean speed in first/last quarter (<0.05 vs >0.12 m/s). Dominant directions do not guarantee exclusively one-direction motion; a full crouch cycle is labelled by its larger ascent/descent path (ties use rise). This method cannot prove all directions have enough examples.

## Budget and diversity

Default quotas, scaled proportionally for a different `--budget-hours`:

| Primary behavior | Hours |
|---|---:|
| single_arm | 6 |
| both_arms | 4 |
| locomotion | 5 |
| yaw | 3 |
| crouch | 3 |
| bend | 3 |
| still | 3 |
| mixed | 3 |

Primary time is counted only once; auxiliary labels may overlap. Categories with low candidate supply relative to quota are sampled first to protect scarce data from overlap conflicts. Within a category, prefer underrepresented subtype, then the dataset with least selected time, then underrepresented amplitude/speed/start-pose bin. Ties use deterministic ordering and seed20261115 candidate shuffling. This is greedy stratified coverage, not a globally optimal diversity solution or a guarantee of equal subtype hours when supplies differ.

Selected windows must not overlap within a source clip. Adjacent windows are allowed and merged into union intervals. Each cached source clip contributes at most60s (`--clip-cap-seconds`). Quantized 17-sample trajectory fingerprints must be unique: joints0.05rad, root-relative path0.02m, relative heading0.05rad, with duration included. This catches exact quantized duplicates, not all perceptually similar motions. Root absolute XY/heading are removed; root height and joint posture are retained. Separate cache clips from one original recording may still pass; original AMASS recording provenance is missing.

Use whole approved windows only: no trimming, repeated segments, reclassification or quota reallocation to fill a deficit. Consequently output may be less than30h, even when some other categories have ample data. Report shortfalls explicitly. Source datasets are not forced to equal hours: LAFAN has little data and only40 long source clips, so a60s-per-clip cap further limits its contribution to <=40min.

## Output and use

- `allowlist.json`: combined index with absolute paths to the three dataset manifests and summary.
- `boneseed.allowlist.json`, `lafan.allowlist.json`, `amass.allowlist.json`: accepted clip IDs, local windows, merged intervals, primary/subtype labels, auxiliary labels and diversity bins.
- `candidates.jsonl`: extracted labels/features for all stage-2 windows, allowing selection-only reruns via `--reuse-candidates /path/to/candidates.jsonl` (same stage2 inputs and rules required).
- `config.json`, `summary.json`, `REPORT.md`: parameters, source/category/subtype hours, deficits and validation.

Intervals are clip-local `[start,end)` in the referenced processed50Hz dataset. NPZ slices use `qpos[starts[clip_id]+start:starts[clip_id]+end]`; AMASS memmaps use meta_motion starts likewise. Keep individual windows for their labels and audit guarantee; merged intervals are the same selected-frame union, not newly evaluated windows. No training config has been switched automatically. This schema is not the legacy exact-segment `generate_dataset.py` format.

This is a **training candidate selection**, not a split-certified dataset or proof of compliance-response coverage. Use a separate recording-disjoint evaluation set when provenance is available; keep the broader stage-1 evaluation pool unchanged. Runtime training sampling must respect the intended time weights—uniform sampling of entire source clips would undo the balance. Force-policy closed-loop evaluation is still required.
