# Online Jacobian EE/root3 residual compliance

Task: `G1/G1_ceer_online_ee_root3_compliance`.
Launcher: `bash scripts/train_ceer_online_ee_root3_residual.sh` (W&B online by default).
Set CUDA_VISIBLE_DEVICES, NUM_ENVS, TOTAL_FRAMES and OUTPUT_DIR explicitly for a full run.

Reuses the old nine-point JacobianComplianceCommand with three zero-offset link
origins: pelvis, left_hand_mimic, right_hand_mimic. The root3 motion dataset is
retained, but offline force caches/variants are not loaded. Frozen CEER and the
integrating residual actor are unchanged. Current action-rate reward indexing
fix applies to new training.

At each control step:
- Joint prompt: 80*(q_next-q_ref) + 20*(q_ref-q_actual).
- Root translation prompt: 60*(p_next-p_actual) + 8*(v_ref-v_actual), norm capped at 40 N.
- Root yaw prompt: 8*wrapped(yaw_next-yaw_actual) + 1.5*(wz_ref-wz_actual), capped at +/-4 Nm.
- Fit world-frame point forces to this generalized prompt using actual-state
  point Jacobians: minimize ||W*(J^T F-target)||^2 + ||F||^2 + ||F-F_previous||^2.
  Weights: root translation 2, rotation 2/0.3, joints 1/0.3.
- Project out opposing net translation/yaw directions, then uniformly scale
  forces to per-point Euclidean norm <=30 N and sum of norms <=60 N.
- Applied forces slew at <=150 N/s per point on physics substeps. The 40 N and
  4 Nm values above bound requested root intent, not achieved wrench.

Every 1-3 seconds select 1-3 active points, weighted by 0.03+reference point
speed. If relative reconstruction error exceeds 0.85, allow up to two retries
using all three points; keep improvements. Fade the old forces to zero before
switching points. Full motion playback continues across these switches.

No external pure torque is applied; turning comes from point force moments.
The pelvis origin alone has zero moment arm about the root origin. These are
soft prompt-fitting objectives, not exact dynamics or a guarantee that every
motion is explainable by three points. Reference targets are used by the force
generator and training reward/critic, not fed directly to the residual actor.
