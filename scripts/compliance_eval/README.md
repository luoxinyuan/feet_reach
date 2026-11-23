# Compliance evaluation v1.5

Two independent evaluations, no combined score. The simulator worker never calls reset after the initial reset. Actor execution uses `get_rollout_policy('deploy')`; the saved phase chooses GT-force versus estimated-force input. Privileged critic tensors remain in the environment but are not actor inputs. All normalization is frozen.

## Run

Use the IsaacLab-enabled gentle Python environment. Select **idle physical GPUs** explicitly; the launcher does not claim busy cards automatically.

```bash
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/compliance_eval/run.py \
  --checkpoint /absolute/path/checkpoint_final.pt \
  --manifest cfg/compliance_eval_v1_5.json \
  --output artifacts/compliance_eval/EXPERIMENT --gpus 0,1,5,6 --videos
```

The runner schedules three tracking jobs and two behavior jobs concurrently. Each tracking job has 160 parallel environments. Behavior has two separate 345-environment rollouts: forced and zero-force. Identical world layouts avoid tiled-position numerical differences. Fewer GPU IDs queue jobs automatically. Results are resumable; use a fresh output directory for another checkpoint. Video-only continuation: add `--video-only`.

For checkpoints with a supported native force command, video evaluation also runs the seven-case motion suite below. The runner detects supported policies from the checkpoint's saved task configuration. Use `--motion-suite` to request the suite explicitly when that metadata is unavailable, or `--no-motion-suite` to skip it.

## Motion evaluation for `new video/`

Run one fixed Crocoddyl reference per direction or action. Reuse the same references for every checkpoint. The reference amplitudes are the largest validated targets currently shared by this suite. The measured maximum is the robot's achieved amplitude within that reference rollout.

| Test | Reference target | Maximum measured response |
|---|---:|---|
| Forward and backward bend, separately | 45° each | Signed change in world torso pitch from the initial pose, positive forward |
| Left and right bend, separately | 45° each | Signed change in world torso roll from the initial pose, positive right |
| Crouch | 23 cm | Drop in pelvis world height from the initial pose |
| Left and right turn, separately | 60° each | Signed change in unwrapped world root yaw, positive left |

Take the peak in the commanded direction for each rollout that survives all 6 s. Report a fall time and pre-fall peak for a failed rollout and leave its completed maximum blank. Also report the signed final value to show how much of the peak was held and record pelvis and ankle XY drift. Keep left and right outcomes separate.

The reference holds for 1 s, moves for 3 s, and holds the endpoint for 2 s. Generate the prompt with the checkpoint's native force rule and run the deployed actor. Latent-curriculum policies use full autonomy. Incremental policies start from a fixed standing latent, while direct-latent policies use their full output. Preserve the ghost reference and applied-force arrows in each of the seven videos under `new video/`. The numerical results go to `crocoddyl_suite_summary.json` and the motion section of `REPORT.md`.

To run just these seven cases, call the suite directly. Use a fresh output folder to keep the seven standard videos together when earlier evaluations contain a larger sweep.

```bash
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/compliance_eval/run_crocoddyl_suite.py \
  --checkpoint /absolute/path/checkpoint_final.pt \
  --output artifacts/compliance_eval/EXPERIMENT \
  --reference-eval artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5 \
  --forward-manifest artifacts/compliance_eval/20261121-surface5-native-spring-forward-bend/manifest.json \
  --gpus 0,1,5,6
```

Cached-force checkpoints require matching prompts for the seven generated references and a worker adapter for the cache format. The aligned 8KP training cache covers the training dataset. Its generated-reference evaluation requires separate adapter support.

## Fixed manifest

`protocol.py --output ...` creates a manifest **once** and refuses to overwrite it. The v1.1 manifest has 128 validation clips and 32 training diagnostics, each 251–1000 frames at 50 Hz. Clips come from stored dataset intervals; full selected intervals up to 1000 frames are replayed. Three fixed seeds repeat each clip. Selection uses motion features, never policy performance. File paths, clip index, source motion/frame IDs, features, category counts and source metadata hashes are recorded.

The available validation dataset has only 3 left-arm-dominant clips and no qualifying stationary clips. Do not pretend the categories are balanced: report counts. Training diagnostics include stationary clips; B provides zero-force standing. Categories are heuristic: walking is screened by root translation, bending by waist excursions, etc. They require human review for semantic claims. Crouch augmentation contains AMASS and BONES data.

Validation means the dataset's stored split. Original AMASS recording IDs are unavailable, so recording-level disjointness is **not established**. The frozen low-level's exposure must also be audited before claiming end-to-end unseen motions. `train_diagnostic` exposure applies to the first tested checkpoint; re-audit for other checkpoints, without changing the selected motions.

A retains the checkpoint's original guidance generator and reference preprocessing (e.g. stored z offset, clamp settings), with initial pose/velocity taken from the reference, initial-state noise disabled. Dataset start and raw motion IDs are asserted. Selected clips are loaded into the usual dataset buffer, retaining canonical raw frame IDs for offline force lookup. Offline-cache policies must explicitly supply a matching cache to the worker; missing coverage must fail rather than silently changing forces. Current launcher targets online generators; use the worker's cache option for an audited offline adapter.

## Fixed-force behavior

23 cases: either hand in six directions (12), pelvis in four horizontal directions (4), pelvis downward (1), shoulder yaw couples (2), torso/pelvis bending couples (4). Hand, pelvis-horizontal, yaw, and bend cases use 5/10/20 N per loaded point; crouch uses 10/20/30 N. Each case/amplitude has five fixed initial perturbations (joint std 0.003 rad). Initial heading is world +X. Hand and pelvis translation directions stay fixed in world relative to the initial heading. Shoulder yaw-couple directions and torso/pelvis bending-couple directions use torso-local transverse axes, transformed by the full torso orientation every physics substep. They therefore follow torso yaw, pitch, and roll, with the bend directions remaining perpendicular to the torso's longitudinal axis. Application points follow their body frames. All offsets are zero except the bending upper point, torso local [0,0,0.25] m; the opposing point is the pelvis origin. This avoids confusing the torso link origin near the waist with the upper chest. No direct torques.

Timeline: 0–2 s zero, 2–3 s cosine ramp up, 3–7 s hold, 7–8 s cosine ramp down, 8–12 s released. Forces and body-attached positions are updated every physics substep. No additional policy-dependent slew filter changes the stimulus. Maximum single-point slew is below 32 N/s, within the training limit of 150 N/s. Each forced/zero pair shares initial state and world placement. Observation noise is disabled for B, actuator alpha is fixed to its configured midpoint, delay to its configured integer midpoint, boot delay is synchronized, and alpha jitter is disabled. Initial states and the complete 0–2 s pre-force root trajectories are checked. A retains the checkpoint’s observation noise and actuator settings.

Motion progression, motion rewards and motion termination are disabled for B; force/history buffers still tick. Out-of-interface loads stay at the specified physical location. GT force observations are populated only for the policy's original supported points; unsupported shoulder/torso loads are labeled. Do not interpret those tests as a fair test of prompt decoding for a three-point GT actor. An estimated-force actor may respond indirectly through proprioception.

## Metrics

Physical failure is root height <0.30 m, root tilt >75 degrees or nonfinite joints. The lower height threshold avoids counting legitimate crouching as a fall. Metrics after first failure are excluded, and failures are always counted. Simulator environments may continue stepping for vectorization, but cannot re-enter the score. A completion means physical survival to clip end, **not** accurate tracking. A separate frame fraction uses joint MAE <0.5 rad and root XY error <0.5 m. Training-specific cumulative-error termination is not used, because it differs between policies.

A: full/group joint MAE, world/root-relative EE errors, root XY/Z/yaw errors, stationary-arm motion, per-point peak/summed force magnitudes, force slew, active-point count. Trial mean and p90 are saved; aggregate summaries give equal weight to trials rather than long clips. Stationary-arm mask is reference max joint speed <0.05 rad/s.

B: survival, paired baseline-corrected response at release, sustained response latency (>2 cm or 3 degrees for 0.1 s), cross-coupling, knee response, release drift and final speed. Coarse directed response means survival plus >5 cm or >10 degrees at release; it does not penalize overshoot and is not a comprehensive success criterion. Failed trials get null endpoint metrics. EE displacement is root-relative. Bending measures torso orientation relative to pelvis; yaw is root world yaw. Non-target response is opposite-hand displacement for EE, yaw degrees for locomotion/crouch, root XY drift for yaw/bending. Read units by group.

## Storage and reproducibility

No frame lists or full simulator dumps. Metric traces are compressed NPZ. Representative videos: three reference clips selected by category and median duration **before inspecting policy results**; one yaw case; all four bend directions at 10 N; hand and root-forward cases; and crouch at 10, 20, and 30 N. 960×540, 25 fps, H.264 CRF 28 with 1.2 Mbit/s maxrate. Tracking videos have mesh ghosts and force arrows. Behavior videos have arrows and a phase overlay.

Each result records checkpoint/manifest hashes, resolved configuration, actor inputs, force-history offsets, physical input bodies and wall time. Worker logs provide startup and simulation timing. `report.py OUTPUT` rebuilds JSON/Markdown summaries. `timing.json` records measured parallel wall time. Source changes are not silently committed or pushed.

Checkpoint loading validates every deployed actor key and tensor shape before evaluation; actor mismatches stop the run. Compatibility skips are limited to critic-only tensors whose shapes changed, because the critic is not queried during rollout. Only actual `DistributedDataParallel` wrappers may be unwrapped; `TensorDictModule.module` is a real layer and must be preserved.

Reference-only `OfflineForcePromptCommand` policies need trajectory-matched force prompts for tracking. Build them from the fixed manifest with `scripts/offline_force_baseline/prepare_compliance_prompt_cache.py`, then pass the generated `cache_map.json` using `--offline-prompt-cache-map`. The worker checks manifest and dataset hashes and injects the current frame's force tokens. Default behavior evaluation applies physical forces and exposes tokens only at points in the actor's trained interface. Add `--prompt-only-behavior` to disable physical force dispatch and supply supported force tokens to the actor instead; unsupported body/offset pairs receive no token and are labeled in the report. For a fixed-clip diagnostic with the checkpoint's training rewards, terminations, randomization, and episode length retained, add `--preserve-training-sim`. These policies have no force-history input.

## Output layout

Standard tracking and behavior videos share `videos/`. The seven generated motion
videos share `new video/`. Metric and video replay raw records live in `metrics/`,
and simulator logs live in `logs/`. Reports remain at the root.

```text
EXPERIMENT/
├── RESULTS_zh.md / REPORT.md / summary.json
├── timing.json / video_validation.json
├── crocoddyl_suite_summary.json
├── new video/             # seven generated motion rollouts
├── videos/
│   ├── tracking_16.mp4
│   ├── left_ee_forward.mp4
│   ├── root_forward.mp4
│   └── ...
├── metrics/
│   ├── tracking_seed0/     # result.json + traces.npz
│   ├── behavior/
│   ├── behavior_zero/
│   └── video_left_ee_forward/
├── logs/
├── previews/              # optional inspection screenshots
└── framework_snapshot/
```
