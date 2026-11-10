# CEER residual — GT reference Δq control

Actor inputs: the unchanged proprioception/self-generated reference (`policy`) and
29 joint increments `delta_q_gt_ = (q_ref[t+1] - q_ref[t]) / 0.04`, ordered by
`active_adaptation.utils.sonic.JOINT_NAMES`. Fixed scaling is invertible, without
clipping or running normalization. It is NOT `q_ref[t+1] - q_actual[t]` and supplies
no absolute target, root trajectory, phase, force history, or estimator. At clip
end the increment is zero. Observations use the current transition; the next
physics step is rewarded against the next reference frame as in the force baseline.

Only the prompt modality changes relative to the offline-force/no-physical-force
teacher: frozen CEER checkpoint, accumulated residual update (2 rad/s, 0.6 rad
reference bound), dense rewards, critic privilege, packed AMASS/EE/still+crouch
corpus, 0.8/0.2 sampling, reset behavior, PPO and seed remain unchanged. Actor and
critic first-layer widths change with 111 force inputs replaced by 29 Δq inputs.
The metadata of the force cache validates the identical motion corpus; force
arrays are not loaded. No physical force or torque is applied. The same `priv`
and `joint_target` remain critic-only.

Launch: `bash scripts/train_ceer_gt_delta_q_teacher.sh`.
Defaults: physical GPUs 5,6,7; 20,480 environments/GPU (61,440 total); 4 billion
frames; W&B online in `ceer-force-prompt-teacher`. The launcher snapshots all source
and saves the frozen checkpoint hash. It starts from scratch.

Validation: `scripts/references/smoke_gt_delta_q.py` runs real Isaac simulation and
PPO: current/next joint indexing, terminal zero, reset alignment, no physical
wrench, actor target isolation, nonzero actor updates, frozen low-level weights,
DDP synchronization, finite losses, checkpoint round trip.

Interpretation: improvement over force tokens identifies losses/optimization in
force encoding as a contributor. Failure also with Δq does NOT establish that Δq
is unusable: this control still has no absolute tracking-error feedback and uses
the same potentially limiting frozen low-level tracker and accumulated residual.
