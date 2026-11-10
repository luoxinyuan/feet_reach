# Independent pelvis + two-hand force-history compliance

Task: `G1/G1_ceer_root3_independent_pelvis`. Launcher:
`bash scripts/train_ceer_root3_independent_pelvis.sh` (physical GPUs 5,6,7 by default).

This experiment inherits the completed force-history task: frozen crouch-trained CEER checkpoint_4500, 29 accumulated joint residual outputs, GT force history offsets [0,1,2,4,8,16], no EE tracking reward, original dense tracking rewards, 80% existing root3 motion corpus and 20% crouch. Actor inputs remain proprioception, its own reference and GT force history; motion targets remain private to force generation/critic/rewards. This is the same GT-force `teacher` phase as the two compared checkpoints, not estimator deployment. No low-level root-interface change is included.

Pelvis target: `F_p = K * (p_ref[t+1]-p_ref[t])` in world coordinates, Euclidean cap 30 N. K=1190.6272466177975 N/m. Calibration sampled 199,383 valid within-clip adjacent frame pairs with the training 80/20 mixture, seed 20261104; 95th percentile displacement 0.01679787 m maps to 20 N, and 0.439% of samples exceed the 30 N raw cap. See `artifacts/independent_pelvis/calibration.json`. No actual root pose/velocity feedback enters this pelvis target. K was calibrated before smoothing/global hand budgeting; quantization of source float16 poses is retained.

Hand solver: actual-pose two-hand Jacobians fit only waist/shoulder/elbow/wrist components of the unchanged joint prompt `80*(q_ref_next-q_ref)+20*(q_ref-q_actual)`. Leg target components are excluded. Joint weights 1/0.3; force and previous-force regularization both 1. No sum-force objective, no roll/pitch moment objective, and no old translation/yaw halfspace projection.

Yaw: wrapped adjacent-reference heading delta, target `clip(83*delta_yaw,-4,4)` N m. The yaw residual weight is `(2/0.3)*sqrt(gate)`, so its squared loss coefficient varies linearly with gate. Gate is smoothstep of reference absolute yaw speed over 0.05–0.30 rad/s. Zero turn intent disables yaw fitting rather than enforcing zero moment. Hand/joint and turning intent can still conflict; logged hand fitting residual includes this compromise.

Fixed semantic channels are pelvis, left hand, right hand. No random dropping or scheduled release. Effective forces may be zero; at most three point forces are applied. Pelvis is capped independently; hand forces are uniformly scaled to respect per-hand 30 N and remaining summed magnitude budget `60-|F_p|`. This preserves pelvis guidance rather than scaling it away when hand demands are large. The inherited common interpolation limiter enforces 150 N/s per point at 200 Hz physics; total magnitude remains <=60 N by convexity. Reset forces/history are zero. No external body torque is applied.

Differences beyond the force formula are intentional: fixed semantic point availability, no release timer, upper-body-only torque fitting, and reference-gated yaw. This is a new allocation design, not a single-loss-term ablation. Residual integration, low-level measured-root interface, rewards and training length are unchanged.

Validation: analytical tests check nonzero pelvis under pure root translation, no force cancellation under stationary-root unilateral hand intent, pelvis independence from hand targets, and force budgets. Simulator smoke checks exercise 3-rank PPO, actor update, frozen low-level, target observation isolation, checkpoint roundtrip and DDP synchronization. Full run uses 20,480 environments/GPU, 61,440 total, nominal 4B transitions (rounded to 3,999,006,720), W&B online.
