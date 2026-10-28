# Direct next-frame root/EE springs

Task: G1/G1_ceer_direct_spring_root3_compliance
Launcher: bash scripts/train_ceer_direct_spring_root3_residual.sh
W&B defaults online. No full training launched as part of implementation.

Three continuously active link-origin springs: pelvis, left_hand_mimic,
right_hand_mimic. At 50 Hz, use next motion-frame world positions (including
scene origins) as anchors and compute F_i=80*(anchor_i-actual_position_i) N.
This is actual-to-next-reference feedback, not a reference-to-reference delta.
No damping, Jacobian solve, global root/yaw allocation, projection, random
contact selection or timed release is used. Turning arises from point forces;
root orientation cannot be independently encoded by a pelvis point spring.

Uniformly scale all forces to <=30 N per point and <=60 N sum of magnitudes.
At 200 Hz physics steps, slew towards those commands at <=150 N/s per point.
Thus applied forces can temporarily differ from instantaneous spring direction.
A single unconstrained spring reaches 30 N at 0.375 m error. Total-budget scaling
can reduce it sooner. Reset starts with zero applied force.

Rewards use the existing CEER MotionTrackingCommand implementation:
- EE position in robot root coordinates: weight 2.
- Root world position / orientation: weights 0.5 / 0.5.
- Root linear / angular velocity: weights 1 / 1.
Other dense tracking and regularization rewards remain inherited. Current-frame
rewards and next-frame anchors follow the existing motion advancement convention.
The reference motion is used for force generation and reward/critic supervision;
the residual actor only receives its own state/reference and applied forces.
Frozen original CEER and integrating residual remain unchanged.

Uses the same motion corpus as online-Jacobian root3, without loading offline
force caches. Includes the action-rate reward indexing fix in the workspace.
This ablation changes both allocation and contact activation (all three always
connected); it is not a contact-sampling-matched comparison. Compare learned
behavior with external/manual prompts separately from assisted tracking.
