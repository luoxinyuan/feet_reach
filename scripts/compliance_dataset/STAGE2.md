# Stage 2: sparse contact expressibility allowlists

This is a lightweight **geometric screening proxy**, not the previous learned-decoder force-information benchmark. It does not certify an injective force-to-motion mapping, physical controllability, or successful tracking. Run after stage 1; it only writes allowlists and audits.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
 /home/xl521/software/miniconda3/envs/gentle/bin/python \
 scripts/compliance_dataset/score_contacts.py \
 --stage1 artifacts/compliance_dataset/20261115_common_quality_v1 \
 --output artifacts/compliance_dataset/new_stage2

/home/xl521/software/miniconda3/envs/gentle/bin/python \
 scripts/compliance_dataset/test_score_contacts.py
```

Use a fresh output directory. `--datasets lafan amass`, `--device cpu`, `--batch 128`, and JSON `--thresholds` overrides are supported. No W&B needed: no policy is trained. Source sizes/mtimes must match stage 1 both before and after scoring.

## Frozen default rules

- Evaluate the stage-1 accepted windows, at eight equally spaced sample times. Reconstruct a 100 ms reference increment; evaluate actual one-frame (20 ms) spring force separately. This samples windows, not every transition.
- Use the G1 USD-exported geometry and a deterministic, dataset-independent atlas of **36 actual mesh-surface points**, two area-weighted samples (triangle centroids) per upper-body mesh part. Include pelvis, torso/head mesh, arms and hands. Exclude legs. This atlas is a numerical proxy, not unrestricted continuous contact optimization. Coordinates are recorded in RULES.json.
- Let `d` contain world root translation, world root rotation vector, waist3 and arm14 increments. Exclude the 12 leg joints because upper-body prompts need not specify a coordinated gait joint-by-joint. This also means isolated leg motions are not assessed by this score.
- Normalize translation by 0.1 m and angles by 0.3 rad: `z = S^-1 d`. The point-displacement map is `A = J S`. There is no learned state prior or decoder.
- Greedily add points that explain the most residual tangent-space energy. Evaluate each prefix of 1 through 5 contacts. For a selected prefix, `y = A z`, `z_hat = A.T (A A.T + 1e-5 I)^-1 y`.
- Overall residual is **squared** error `||z_hat-z||²/(||z||²+1e-8)`, threshold **0.65**. It is not NRMSE or mutual information. It approximately asks for at least 35% normalized tangent-space energy recovery.
- For root translation / root rotation / waist / left arm / right arm, require residual <= **0.90** only when that group's energy exceeds both **0.0025** and **10%** of the total. Very small groups should not dominate ratios. Overall residual still includes errors on stationary dimensions.
- Sample energy below `1e-4` bypasses the information gate. Stage-1 `still` windows also bypass it; neither bypasses force limits.
- At least **80%** of eight samples must pass the information gate (effectively at least 7/8).
- Actual nominal spring force is `80*(FK(q[t+1])-FK(q[t]))` N at the selected material points, including full root/joint motion. Require single-point <= **30 N**, sum of Euclidean point-force magnitudes <= **60 N**, in at least **95%** of samples (effectively all eight). No cap is applied before checking.
- Same-contact force derivative uses the following frame's force; its sampled P95 maximum-point norm must be <= **150 N/s**. It does **not** measure jumps due to changing contact identities.
- Accept a window if **any** prefix up to the budget passes. Prefer the smallest passing contact count; otherwise report the lowest-residual rejected prefix. Max5 contains every accepted max3 window by construction.

## Development correction, not a hidden threshold change

The first LAFAN pilot inverted exact 100ms FK displacements and gated every small moving group. It admitted just 11/4075 windows for three contacts; finite-displacement nonlinearity and inverse conditioning inflated per-group errors. That pilot is retained under `20261115_stage2_pilot`, not used as the final list.

The corrected pilot uses linear tangent projection for geometry and significant-group gating, with exact FK reserved for force budget/rate checks. It is retained under `20261115_stage2_pilot_v2`. The final version also evaluates all smaller prefixes rather than requiring exactly three/five forces. Thresholds 0.65/0.90/80% were not tuned separately by dataset. This development used LAFAN, so these filters are not an independent test of LAFAN performance.

## Files and limitations

`<dataset>.max3.allowlist.json` and `.max5.allowlist.json` retain stage-1 source IDs and clip-local `[start,end)` spans. `*.scores.jsonl` stores selected count, scores, rejection reasons and force diagnostics per window/budget. Summary union hours do not count overlapping windows twice. These are not legacy `generate_dataset.py` exact-segment allowlists.

Selection is an offline **per-sample oracle**: points can differ at each sampled instant. No temporal contact scheduler, release ramp, collision/accessibility constraints, perturbation robustness, learned force decoder, or frozen-tracker rollout is validated. Thus passing is an optimistic sparse geometric capability test. Rejecting means the chosen atlas/search/metric failed; it is not proof that no suitable contact arrangement exists. Root and joint scale choices affect results.

Keep the stage-1 public evaluation pool unchanged. Use these method-biased lists only for training selection/ablation; compare retained hours and category coverage alongside subsequent tracking results. Do not randomly split AMASS cached segments into train/test until original-recording provenance is restored.
