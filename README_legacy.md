# G1 Wall Foot Reach

G1 全身扶墙脚部 reaching：左手扶墙、右脚支撑、左脚跟踪实时 root 坐标系下的笛卡尔目标。

包含 Crocoddyl 参考姿态与连续轨迹生成、沿用 low-level 框架的 teacher/adapt/finetune
训练、target joint estimator、0–20 N 脚底 ramp 外力扰动、固定目标精度评估与网页遥操作。

## 使用入口

- [参考姿态生成](scripts/references/README.md)
- [轨迹数据集、训练与外力配置](scripts/references/FOOT_REACH.md)
- [固定点评估、录像及 VS Code 网页遥操作](scripts/foot_reach/README.md)
- 任务配置：`cfg/task/G1/G1_wall_foot_reach.yaml`
- 八卡三阶段训练：`WANDB_RUN_NAME=v2 bash scripts/train_wall_foot_reach_pipeline.sh`
- 评估录像：`bash scripts/eval_wall_foot_reach.sh --checkpoint /path/to/checkpoint_final.pt --video`
- 网页控制：`bash scripts/eval_wall_foot_reach.sh --checkpoint /path/to/checkpoint_final.pt --web --port 8765`

训练和评估依赖 Isaac Lab / Isaac Sim。运行前请按本机环境调整脚本中的 Python、Isaac Lab
路径；生成参考使用单独的 Crocoddyl 环境，依赖锁定见 `scripts/references/requirements-lock.txt`。
网页默认监听 localhost，可通过 VS Code Remote SSH 的 Ports 面板转发 8765 端口访问。

Git 仓库包含源代码及 G1 USD 模型；`dataset/`、`artifacts/`、`outputs/`、WandB 日志和
训练 checkpoint 不在版本控制中。新机器上需要依照上述文档生成数据，或自行复制数据及 checkpoint。

本项目基于原 Hierarchical Humanoid Compliance 代码扩展，以下保留原框架说明。

---

## Original framework: Hierarchical Humanoid Compliance

This repository provides Isaac Lab training and evaluation code for a
two-layer humanoid compliance controller. A shared low-level whole-body policy
tracks motion references, while a high-level policy generates residual
end-effector or root commands. The repository also includes fixed-compliance,
ranged-compliance, force-estimator, and analytical MoE baselines.

Some checkpoint identifiers may retain legacy names for compatibility with
previous training runs. They do not change the method or the public repository
name.

## Repository layout

```text
active_adaptation/       Environment, commands, observations, policies, and utilities
cfg/                     Hydra task, algorithm, evaluation, object, and MoE configs
scripts/train.py         Training entry point
scripts/eval_manipulation.py
                         Manipulation, EE-compliance, and root-compliance evaluation
scripts/data_process/    Motion conversion and dataset allowlists
train.sh                 Low-level training launcher
train_hl_teacher_student.sh
                         High-level teacher-student launcher
```

## Requirements

- Ubuntu 22.04 is recommended.
- NVIDIA GPU with a driver compatible with the selected CUDA toolkit.
- Isaac Sim 4.5.0.
- Isaac Lab 2.2.0.
- Python 3.10.

The commands below use the same environment variables expected by the
training and evaluation scripts:

```bash
conda create -n humanoid-compliance python=3.10
conda activate humanoid-compliance

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab separately:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
cd -
```

Install this repository:

```bash
pip install -e .
```

Set the Isaac Lab paths before running any simulator command. Adjust the
paths and GPU list for the local machine:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PWD"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PREFIX/bin:$PATH"
```

Before training or W&B-backed evaluation, replace every
`your-wandb-entity`, `your-low-level-project`, and `your-high-level-project`
placeholder with the user's own W&B entity and project. The launchers and the
MoE manifest intentionally contain no private account or checkpoint paths.

Optional checks:

```bash
isaacsim
python -m compileall -q active_adaptation scripts
```

## Dataset download and preprocessing

The motion datasets are not redistributed in this repository. Download the
AMASS/LIMMT data from their official distribution channels and follow their
licenses. Extract the motion files into a local directory containing `.npz`
files.

The converter accepts either a single `.npz` file or a directory tree. It
selects the G1 joints and bodies, converts LIMMT keypoint motion when needed,
resamples to 50 Hz, computes velocities and world-frame body poses, segments
the motions, and writes a memory-mapped dataset.

For the standard 3-keypoint stiff policy, create the dataset expected by the
default stiff configuration:

```bash
cd /path/to/this/repository

python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/AMASS \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

The allowlist contains the selected motion segments and preprocessing
parameters. To convert a different collection, omit `--allowlist` or use the
matching allowlist in `scripts/data_process/`.

The convenience wrapper `generate_limmt_amass_full_dataset.sh` performs the
same conversion. Before using it, edit its `DATASET_ROOT`, `OUT_DIR`, and
`PYTHON` variables for the local machine.

The task config refers to datasets by paths relative to the repository's
`dataset/` directory. If a task refers to a locally generated rollout dataset,
replace `task.command.dataset.mem_paths` in the selected task config with the
processed dataset path before training.

## Low-level training

`train.sh` runs the low-level teacher-student pipeline:

```text
train       4B environment frames
adapt       1B environment frames
finetune    2B environment frames
```

Edit the W&B project, GPU list, process count, and master port at the top of
`train.sh`, then select one supported pipeline through `PIPELINE`:

```bash
# Shared 3-keypoint stiff backbone; maximum external force is 30 N.
PIPELINE=stiff30 bash train.sh

# End-to-end fixed 200xyz compliance baseline.
PIPELINE=fixed_ee bash train.sh

# End-to-end independently ranged xyz compliance baseline over 100--600 N/m.
PIPELINE=range_100_600 bash train.sh

# Optional 5-keypoint locomotion policy.
PIPELINE=5kp bash train.sh
```

The corresponding task configs are:

```text
cfg/task/G1/G1_3kp_stiff.yaml
cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml
cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml
cfg/task/G1/G1_5kp.yaml
```

Each pipeline creates W&B runs for the three stages. The later stages load
the previous stage through `checkpoint_path`. To resume locally, pass a local
checkpoint through the corresponding Hydra override or use the local
evaluation interface described below.

## High-level teacher-student training

Use `train_hl_teacher_student.sh` for high-level EE policies. The low-level
checkpoint is frozen and shared by all high-level policies. The high-level
script runs two stages:

```text
teacher     root_student_force_ppo
adapt       root_student_force_ppo_adapt
```

Set the low-level checkpoint and W&B settings before launching:

```bash
export LOW_RUN_PATH=your-entity/your-low-level-project/low_level_finetune_run
export HL_PROJECT_PATH=your-entity/your-high-level-project
export HL_WANDB_PROJECT=your-high-level-project
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC=4
export MASTER_PORT=29502
```

Train the ten high-level experts used by the current 100--600 N/m analytical
MoE:

```bash
RUN_ONLY=experts bash train_hl_teacher_student.sh
```

The ten expert targets are:

```text
100x, 100y, 100z,
200x, 200y, 200z,
400x, 400y, 400z,
600xyz.
```

Train the matched high-level baselines:

```bash
RUN_ONLY=baselines bash train_hl_teacher_student.sh
```

These are the fixed 200xyz high-level policy and the 100--600 ranged
high-level policy. The corresponding task configs are under:

```text
cfg/task/G1/hl/ee/
```

The root-compliance templates are kept under `cfg/task/G1/hl/root/`. They are
not launched by default; uncomment the desired `run_pipeline root ...` line
in `train_hl_teacher_student.sh` when a root policy is needed.

## Analytical MoE

The analytical MoE is training-free. It freezes the ten expert checkpoints,
interpolates their residual actions in physical displacement space, and
composes the x, y, and z components independently. The current manifest is:

```text
cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml
```

Fill the ten `checkpoint_path` entries with either local checkpoint paths or
W&B run specifications before launching. Then evaluate a commanded stiffness:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --objects cfg/objects/boxes_scene.yaml \
  --moe-experts-config \
    cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml \
  --ee-compliance-stiffness 300 450 600
```

The analytical MoE has no single policy checkpoint and does not support
`--export`; evaluate it directly through the manifest.

## Baselines and evaluation

The evaluation script supports W&B runs, local checkpoints, and the analytical
MoE manifest.

Evaluate a W&B policy:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --objects cfg/objects/boxes_scene.yaml \
  --run_path your-entity/your-project/run_name \
  --ee-compliance-stiffness 200 200 200
```

Evaluate a local checkpoint. Supplying the matching training config avoids a
W&B download and preserves the checkpoint's observation/action definition:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --objects cfg/objects/boxes_scene.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/training/cfg.yaml \
  --ee-compliance-stiffness 200 200 200
```

The principal baseline modes are:

| Baseline | Training or evaluation entry point |
|---|---|
| Analytical MoE | `--moe-experts-config cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml` |
| Stiff.-conditioned high-level | `RUN_ONLY=baselines bash train_hl_teacher_student.sh`, ranged HL task |
| Stiff.-conditioned end-to-end | `PIPELINE=range_100_600 bash train.sh` |
| Fixed end-to-end | `PIPELINE=fixed_ee bash train.sh` |
| Raw stiff low-level | `PIPELINE=stiff30 bash train.sh` |
| Estimated-force analytical compliance | Add `--ee-compliance-force-estimator-ablation` when evaluating the stiff low-level checkpoint |
| Oracle-force analytical compliance | Add `--ee-compliance-oracle-force` when evaluating the stiff low-level checkpoint |

The force-estimator and oracle-force modes bypass the high-level EE residual
output and directly construct a low-level EE target from force divided by the
commanded stiffness. The oracle mode uses the evaluator's applied force; the
estimator mode uses the force predicted from deployable observations.

Example oracle-force evaluation:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --objects cfg/objects/boxes_scene.yaml \
  --run_path your-entity/your-project/stiff_low_level_run \
  --ee-compliance-oracle-force \
  --ee-compliance-stiffness 300 450 600
```

For a full bimanual command, pass six values in the order
`left_xyz right_xyz` and use `--ee-bimanual-compliance-eval`.

## Dummy teleoperation and interactive environments

`teleop_dummy_pub.py` publishes root, head, left-hand, and right-hand poses
over UDP. Run the simulator and the publisher in separate terminals from the
repository root.

### Live teleoperation with `eval_manipulation.py`

Start the policy first. Use `-p` to enter interactive simulation mode and
`--obs_source udp` to consume the UDP pose stream:

```bash
python scripts/eval_manipulation.py \
  --run_path your-entity/your-project/run_name \
  --full_collision \
  --num_envs 1 \
  --obs_source udp \
  --objects cfg/objects/table_scene.yaml \
  -p
```

Then start the dummy publisher in a second terminal:

```bash
python teleop_dummy_pub.py \
  --mode 0 \
  --dst_ip 127.0.0.1 \
  --dst_port 15000 \
  --hz 30
```

The simulator listens on UDP port `15000`. The publisher sends the `G6D1`
packet containing four world-frame poses, each represented by position
`(x,y,z)` and quaternion `(x,y,z,w)`. For live teleoperation, keep one
publisher and one simulator process per port.

The keyboard controls in mode 0 are:

```text
Hands: I/K = x +/-, J/L = y apart/together, U/O = z +/-
Root:  W/S = x +/-, A/D = y +/-, F/H = z +/-, Q/E = yaw +/-
Exit:  ESC or Ctrl+C
```

### Available object environments

Pass one of the following files through `--objects` in interactive play mode:

| Scene | Contents |
|---|---|
| `cfg/objects/controller_test.yaml` | Minimal dynamic-box collision check |
| `cfg/objects/boxes_scene.yaml` | Platform and dynamic cylinder |
| `cfg/objects/table_scene.yaml` | Table, box, and sphere |
| `cfg/objects/room_scene.yaml` | Table, cylinder, and bed |
| `cfg/objects/wall_scene.yaml` | Front and side wall obstacles |

For example:

```bash
python scripts/eval_manipulation.py \
  --run_path your-entity/your-project/run_name \
  --full_collision \
  --obs_source udp \
  --objects cfg/objects/boxes_scene.yaml \
  -p
```

`--num_envs 1` is recommended for interactive use. With multiple environments,
the same incoming UDP target is broadcast to every environment. The object
scene is loaded only in interactive play mode; scripted EE/root compliance
evaluation clears objects so that external-force measurements remain isolated.

### Dummy publisher modes

| Mode | Command | Purpose |
|---|---|---|
| `0` | `python teleop_dummy_pub.py --mode 0` | Keyboard-controlled fixed pose and root motion |
| `1` | `python teleop_dummy_pub.py --mode 1` | VR pose input; the right-controller B button toggles default pose/VR control |
| `2` | `python teleop_dummy_pub.py --mode 2` | Random hand pose every five seconds for smoke testing |
| `3` | `python teleop_dummy_pub.py --mode 3` | Receives root-state CSV on UDP port `15001` and republishes a pose packet to port `15000` |
| `4` | `python teleop_dummy_pub.py --mode 4 --motion-file /path/to/motion.npz` | Replays a motion with optional foot positions |
| `5` | `python teleop_dummy_pub.py --mode 5 --motion-file /path/to/motion.npz` | Replays a motion and appends joint targets |

Modes 0--2 are the usual choices for testing a high-level policy with
`eval_manipulation.py`. Mode 1 requires the local VR interface. Mode 3 needs a
separate process that publishes CSV lines in the format
`timestamp,x,y,z,qw,qx,qy,qz` to UDP port `15001`.

Modes 4 and 5 are intended for motion-tracking receivers that support the
additional packet fields. The simple `TeleopCommand` receiver used by
`cfg/task/G1/G1_manipulation.yaml` accepts only its own legacy 28-float packet;
use a checkpoint/configuration based on `MotionTrackingCommand` for the
extended replay modes.

### Using a local checkpoint

The same interactive command works with a local checkpoint when its matching
training configuration is supplied:

```bash
python scripts/eval_manipulation.py \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/training/cfg.yaml \
  --full_collision \
  --obs_source udp \
  --objects cfg/objects/controller_test.yaml \
  -p
```

Do not add `--ee-compliance-eval` or `--root-compliance-eval` when using dummy
teleoperation. Those modes replace the live UDP command with a scripted
evaluation protocol and remove objects from the scene.

## Evaluation protocol and outputs

The EE compliance evaluator uses six force directions (`+/-x`, `+/-y`,
`+/-z`) and force magnitudes of 5, 10, 15, 20, and 30 N. Each probe contains
a warmup, a force ramp, a constant-force hold, and a recovery phase. Apparent
stiffness is computed from the stable force and displacement windows.

Useful commands:

```bash
# EE tracking with scripted target poses.
python scripts/eval_manipulation.py \
  --run_path your-entity/your-project/run_name \
  --ee-tracking-eval \
  --num_envs 8

# Root compliance evaluation.
python scripts/eval_manipulation.py \
  --run_path your-entity/your-project/run_name \
  --root-compliance-eval \
  --root-compliance-num-envs 8
```

Reports are written as JSON files under `outputs/` by default. Use
`--ee-output` or `--root-output` to choose an explicit output path. The JSON
reports retain per-probe force, pose, stiffness, error, and stability data so
that summary tables can be regenerated without rerunning the simulator.

## Reproducibility notes

- Use the same task config, low-level checkpoint, force protocol, number of
  environments, and stiffness command when comparing policies.
- Set the random seed through the Hydra `seed` override when a deterministic
  evaluation seed is required.
- Keep each high-level checkpoint matched to the low-level checkpoint used
  during training.
- Use the checkpoint's matching VecNorm. Analytical MoE experts retain their
  own checkpoint-specific VecNorm and do not apply a second shared
  normalization.
- The default evaluation mode disables training-time TorchInductor compilation
  and uses eager inference to avoid long first-rollout compilation delays.

## License and external data

Check the repository license and the licenses of all external assets and
datasets before redistribution. AMASS/LIMMT data and any downloaded model
checkpoints remain subject to their original terms.
