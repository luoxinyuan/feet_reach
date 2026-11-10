# Online Jacobian root3 with force adaptation

Task `G1/G1_ceer_online_estimated_root3`; launch
`CUDA_VISIBLE_DEVICES=2,3,4 NUM_ENVS=16384 bash scripts/train_ceer_online_estimated_pipeline.sh`.
W&B defaults online. Per-stage checkpoints and automatic phase transitions.

Force generator/data/rewards inherit the original 3kp online configuration,
including EE tracking (this is not the separate nine-point no-EE experiment).
The frozen general CEER low-level and integrating 29-joint residual are unchanged.
No offline force cache is loaded. Canonical force channel order is pelvis,
left_hand_mimic, right_hand_mimic, irrespective of sampled active-slot order.

Three phases reuse DenseForcePPO:
1. Teacher 4B frames: public robot/self-reference state (354) + 9 GT forces /30.
2. Distill 1B: student rollout; teacher frozen; residual action imitation and
   force MSE. Estimator MLP 930 -> 512 -> 256 -> 9, from 10 causal frames of
   root angular velocity/gravity, joint position/velocity, actual low-level action.
3. Finetune 4B: estimated-force actor PPO with dense rewards, imitation weight .05,
   estimator updates after PPO. VecNorm frozen in phases 2/3.

Public deployment student in_keys are policy and estimator_history_ only.
GT forces (actual applied forces after slew), future reference and privileged
state can supervise rewards/critic/estimator but do not enter deployed actor.
Estimator initializes from scratch; no pre-trained CEER2 estimator weights are
claimed. The existing GT-only 3kp online checkpoint is not loaded: a new GT
teacher uses the staged input/normalization layout to keep phase transitions
identical to the tested three-stage pipeline.

Reset clears history; no target joint state or measured contact force appears in
estimator input. Force magnitudes are limited to 30 N per point in prediction.
The frozen CEER's internal state estimator is separate from this force estimator.
Evaluation must compare GT teacher, estimated student, force error and manual
interaction: success with a GT teacher does not establish force identifiability.
