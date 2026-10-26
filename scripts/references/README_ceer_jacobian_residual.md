# CEER residual: hybrid Jacobian force-generator ablation

This experiment changes the guidance generator relative to
`outputs/ceer-residual/full_20261022_144917`. It does not change the frozen original
CEER checkpoint, bypassed joint predictor, residual integration, public actor
observations, privileged critic, dense rewards, training split or PPO settings.
It trains a NEW residual actor from scratch, not a fine-tune of the old student.

## Hybrid intent

At each 50 Hz control frame, for bounded G1 joints in articulation order:

```
tau_joint = 80 * (q_reference_next - q_reference_current)
          + 20 * (q_reference_current - q_actual)
```

These are virtual prompt moments, not inverse dynamics. Reference sampling is
fixed at 50 Hz; the increment gain is defined for this interval. Set
`task.command.joint_feedback_gain=0` for a joint reference-increment-only ablation.
This does NOT disable the root's feedback: root translation and yaw keep the
original spring/damping intent (60 N/m, 8 N s/m; 8 N m/rad, 1.5 N m s/rad;
40 N and 4 N m intent caps). Desired root roll/pitch moment is zero. No explicit
torque is applied to the robot. The optimizer produces only 1–3 point forces.

The Jacobian is evaluated at the ACTUAL robot configuration. PhysX's world-frame
COM Jacobian is shifted to the link origin using `v_point=v_com+omega×offset`.
Root columns are expressed about the root-link origin. Full generalized vectors
have 6 floating-base entries plus 29 joints. Root moment and joint entries are
normalized by a 0.3 m characteristic arm; root rows have an additional weight 2.

Minimize `||W(J.T f - tau)||² + ||f||² + ||f - f_previous||²` using a batched
9-by-9 solve. Inactive force slots are masked. As in the baseline, project away
opposition to net translation/yaw intent, uniformly scale to 30 N per point /
60 N summed magnitudes, then slew actual force at <=150 N/s per physics substep.
The actor observes these actual post-smoothing forces, not optimizer targets.

Points are the same nine upper-body link origins as the baseline. Keep each
contact layout for 1–3 seconds, fade all forces to zero before switching. On
selection, allow two retries when post-cap normalized reconstruction error
exceeds 0.85; retry with three points and keep the best encountered layout.
Fallback is that best approximation; no clips are discarded or relabeled as
exactly representable. This resampling differs from the old heuristic and is
part of the generator ablation, not an isolated test of just the matrix solver.

Upper-body point forces cannot encode arbitrary leg joint moments. Full-target
reconstruction error may remain large. Log actual and desired weighted residual,
relative error and cosine, plus an unbounded, nearly unregularized projection
for the SAME active points to distinguish geometric limitations from force caps
and smoothing. The unbounded projection is diagnostic only and NEVER applied.
This hybrid is deliberately not an exact reproduction of the proposal's pure
reference-frame-increment generator.

## Verification and training

- `tests/test_jacobian_guidance.py`: recover known representable generalized
  loads, virtual-work identity, inactive/degenerate contacts, COM shift sign.
- Existing force envelope tests: per-point/total cap, slew and nonopposing intent.
- `scripts/sonic/smoke_train.py --backend ceer --jacobian`: real PPO, frozen CEER,
  target poisoning, per-env reset isolation, three-rank synchronization, force
  envelope and fade, and Jacobian-predicted versus measured point velocity at
  1e-4 tolerance. Output: `artifacts/ceer_jacobian_residual/`.

```bash
CUDA_VISIBLE_DEVICES=1,2,4 NUM_ENVS=16384 TOTAL_FRAMES=4000000000 \
  bash scripts/train_ceer_jacobian_residual.sh
```

Keep the original 49,152 global environments, 32 rollout steps, 5 epochs and
8 minibatches: 2,543 updates / 3,999,793,152 actual frames. Use a run-local frozen
checkpoint copy and save source/config snapshots for the full experiment.
Compare on the original held-out clips and initialization, including no-force
standing, success, joint/keypoint/root error and force usage; report this as one
seed rather than evidence of statistical superiority.

## Full experiment launched 2026-10-23

- Run directory: `outputs/ceer-jacobian-residual/full_20261023_131021`.
- Background PID: 1204407; physical GPUs 1, 2, 4, 16,384 environments each.
- Frozen CEER is a run-local immutable copy with the original checkpoint SHA256
  `637ff52b2712b4ad3c5fa4b8004a85ff11567e703a92532345867de28c45dc7c`.
- `source_snapshot/`, `source_hashes.json`, `.hydra/` and `launch.json` record
  sources, resolved config, command and throughput estimate.
- Eight-update benchmark: steady median 122847 frames/s,
  approximately 31–32 GiB device allocation per GPU (PyTorch peak 22.75 GiB).
- Expected duration about 9 hours; approximate finish Oct 23 22:15 EDT /
  Oct 24 10:15 Beijing time. Allow 8–10 hours as utilization changes.
- `train.log` has metrics; checkpoints every 50 updates under `wandb/*/files/`.

Three-rank smoke tests each completed two PPO updates. Seven pure tensor tests
passed. A resolved-config comparison confirmed identical algorithm, action,
observations, rewards and dataset settings between baseline and this variant.
Initial random-policy diagnostics have full generalized-load relative errors
around 0.94–0.96; even unbounded projection errors are around 0.91–0.92. These
are NOT tracking errors or trained-policy scores; unsupported leg components
and coupled force directions are substantial. Evaluate final behavioral
tracking and human-guidance usability, not just optimizer convergence.
