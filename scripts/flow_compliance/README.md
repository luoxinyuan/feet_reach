# Force-conditioned latent flow / ComplianceMix

This experiment freezes released SuperMimic and trains two separate networks:
1. A causal MLP PPO teacher collects on-policy experience (privileged critic only).
2. A conditional flow learns one 32-D latent-residual endpoint from that
   experience. The flow is the deployed model; it is not optimized with an
   incorrectly assumed Gaussian PPO likelihood.

Inputs: normalized proprioception/self-generated reference, and GT **applied**
contact force/location history (offsets 0,1,2,4,8,16). No future motion/force GT,
no estimator. Flow supervision consists of the current actually sampled PPO
latent action (bounded with tanh), not GT trajectory labels. Reset and terminal samples are excluded. Capped exponential normalized advantage weights
favor better experience; these are heuristic distillation weights, not an
unbiased policy gradient or a guarantee of successful behavior. Fifty PPO
updates warm up the teacher before flow training. Both continue training.

Flow: MLP condition encoder, conditional rectified-flow velocity MLP, Gaussian
noise -> bounded latent endpoints; uniform flow time and standard conditional
flow matching MSE. Eight Euler steps at inference. One 32-D endpoint is sampled at each 50 Hz control step. No future action
labels or action chunking are used. SuperMimic uses its
existing causal endpoint smoothing (1/10 remaining difference each step),
reference-free hold anchor and frozen decoder. This is not exact finite-horizon
linear interpolation. Stochastic deployment keeps the noise seed per episode;
MODE uses fixed-seed Gaussian noise for reproducible sampling, not the true
distribution mode. Persistent noise reduces mode switching but does not guarantee temporal
consistency. Multimodality cannot be manufactured if the teacher never explores
alternative useful strategies.

Data: `dataset/compliance_mix_29h_v1`, 29.224 hours from BONES Seed, LAFAN, AMASS.
Online upper-body continuous-surface force generator: at most 5 contacts,
80 N/m spring to next reference surface point, 30 N/contact, 60 N sum of norms,
existing force slew and contact scheduling. Dataset and low-level weights stay
unchanged.

Run: `bash scripts/train_supermimic_flow_latent.sh` (physical GPUs 5,6,7).
Override `NUM_ENVS_PER_GPU`, `TOTAL_FRAMES`, `OUTPUT_ROOT` via environment.
W&B online; both teacher and flow plus all optimizer states checkpointed.
Evaluation/deployment: `agent.get_rollout_policy('eval'/'deploy')` returns FLOW;
`get_rollout_policy('train')` returns PPO teacher. Training reward curves therefore
measure the teacher, not flow closed-loop performance. Evaluate the flow
separately after training; small flow MSE alone does not prove compliance.

Validation: `scripts/flow_compliance/smoke.py` checks actual physics, no future
reference input, force budgets, frozen low-level, teacher+flow parameter updates,
three-rank synchronization and checkpoint roundtrip. Smoke W&B is disabled.
The fixed-budget main run uses 4e9 environment transitions, as existing baselines.

The single-latent checkpoint format is `single_latent_v1`; loading the former
ten-step checkpoint is explicitly rejected. The 10-step smoothing constant is
unchanged and does not denote ten output latent vectors.
