# CEER residual force-prompt whole-body compliance

## Architecture

Uses the original general CEER finetune `.cache/ceer/checkpoint_final.pt`
(SHA256 `637ff52b2712b4ad3c5fa4b8004a85ff11567e703a92532345867de28c45dc7c`).
The original `actor_student` and privileged-state estimator `adapt_module` are
frozen. `adapt_joint_module` is **not instantiated**: the actor receives the
student's self-generated joint reference directly, normalized with the original
checkpoint's fixed ObservationNorm statistics.

At 50 Hz, the residual actor outputs 29 values and integrates
`q_ref += 0.02 * 2 * tanh(action)`, with joint limits and a 0.6 rad tracking-error
bound, identical to the SONIC experiment. Reset starts from measured joint angles.
The original five-frame joint interface receives five copies of the held `q_ref`,
in the original AMASS joint order (not Isaac articulation order). No future
AMASS frame is sent to either actor.

CEER still requires its root/wrist interface. Wrist poses are computed by batched
forward kinematics from this same `q_ref` and the simulation USD. Root height and
local XY velocity come from measured state; heading means current heading. Thus
there is no independent desired root velocity/yaw command in this experiment;
locomotion depends on learned joint residuals and physical prompting. This is a
causal adapter, not a claim that CEER and SONIC have identical capabilities.
Original CEER motor parameters, action scaling, filtering, delay and boot
protection are retained.

The CEER force-conditioning channel is fixed to **5**, near its saved input mean.
This is distinct from the **30 N per-point physical guidance cap** (60 N combined),
which is unchanged. In an 8-environment diagnostic, setting this conditioning
channel to 30 gave elbow MAE 0.170 rad; 5 gave 0.052 rad with stable standing.

Student observations contain proprioception/history, actual upper-body forces,
previous residual actions, and its own reference/error/update velocity. They do
not contain the dataset target. Dense AMASS targets are used by rewards and the
privileged critic only, and to generate the physical force prompt. Initial states
are sampled from the dataset as in SONIC.

Task/data/rewards/force generation/PPO inherit the SONIC configuration: AMASS
LIMMT full processed training data, 1–3 upper-body force application points,
smooth forces, no direct applied torque. Internal observation names
`sonic_reference` and `sonic_joint_velocity` are shared generic terms; they do not
load or invoke SONIC in this CEER task.

## Validation

- `scripts/ceer/check_native.py`: direct-reference output versus the original
  exported actor: max absolute error 1.31e-6; batch versus single 4.77e-7.
- `scripts/sonic/check_standing.py --backend ceer`: 8 robots, 10 s standing with no
  resets, minimum root height 0.750 m, maximum tilt 0.151 rad; another 6 s of
  sinusoidal elbow reference tracking gives 0.0524 rad MAE.
- USD FK versus actual simulated wrist transforms: position error < 0.000001 m,
  orientation error < 0.000002 rad.
- `scripts/sonic/smoke_train.py --backend ceer`: real PPO updates, frozen weights
  unchanged, actor updates, target poisoning leaves public inputs/actions
  unchanged, per-environment reset isolation. Three-rank test on physical GPUs
  1,2,4 with 2048 environments each passes synchronized parameters and LR checks.
- Logs and reports: `artifacts/ceer_residual/`. These checks establish correct
  wiring and training execution, not the eventual learned compliance quality.

## Launch

```bash
CUDA_VISIBLE_DEVICES=1,2,4 NUM_ENVS=16384 TOTAL_FRAMES=4000000000 \
  bash scripts/train_ceer_residual.sh
```

The three GPUs synchronize one residual policy. Same global batch and requested
4 billion frames as SONIC: 49,152 environments, 32 rollout steps, 5 PPO epochs,
8 minibatches. Full batches yield 2,543 updates / 3,999,793,152 frames. Frozen
CEER is outside all optimizers. Keep its checkpoint available when loading a
residual checkpoint. Launcher records its hash in each output directory. Relative checkpoint paths
resolve against the repository root, including when Hydra changes directories.

## Full run started 2026-10-22

- Physical GPU indices: **1, 2, 4**; 16,384 environments each.
- Directory: `outputs/ceer-residual/full_20261022_144917`.
- Background launcher PID: `248301`; started 14:49 EDT.
- Frozen checkpoint copied into this run as `frozen_ceer.pt`, and the resolved
  training config points to that immutable run-local copy. Adapter sources also
  saved under `source_snapshot/`.
- 8-update benchmark while SONIC also ran: steady median **215,525 frames/s**;
  approximately 31 GB total device memory per GPU (22.75 GB PyTorch peak).
- Estimated training time ~5–6 hours, roughly Oct 22 20:00–21:00 EDT / Oct 23
  08:00–09:00 Beijing time, subject to shared-machine throughput.
- `train.log` contains live metrics; checkpoints every 50 updates, under the
  W&B offline run's `files/` directory. `latest_full_run.txt` points to this run.
- Startup verified through 3 PPO updates / 4,718,592 frames: finite losses,
  KL 0.00570, physical per-point peak 30 N and combined budget 60 N. Initial
  training joint MAE 0.259 rad is an early rollout statistic, not a final eval.
