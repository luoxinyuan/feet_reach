# G1 扶墙左脚 reach 数据集与 low-level 训练

固定左手扶左侧墙、右脚支撑，保持已确认的躯干/手臂/支撑腿姿态，仅改变左腿六个关节。目标点是**左脚脚底中心**，不是 ankle link 原点；root 坐标使用完整基座姿态（不是仅 yaw）。

## 已生成的数据

- `artifacts/foot_reach/manifest.json`：生成参数、约束、拒绝原因、训练/验证划分。
- `artifacts/foot_reach/workspace.npz`：候选可达点、完整配置、通过连续轨迹验证的 endpoint IDs。
- `artifacts/foot_reach/motions/{train,val}/*.npz`：完整 FK、关节位置、解析速度/加速度、接触力与力矩。
- `dataset/wall_foot_reach/{train,val}`：**现有 AMASS 转换后使用的 MotionData memmap schema**，29 个关节、27 个刚体，50 Hz，float32。它不是原始 AMASS/SMPL 参数格式。
- `source_files.json`：每个 memmap motion 对应的源文件；不依赖文件系统遍历顺序。
- `artifacts/foot_reach/trajectory_preview.html`：真实 USD 网格的轨迹动画；`workspace.html` / `workspace.png`：root 系工作空间。

274 个笛卡尔候选点中有 74 个通过筛选；每点两种速度/停留版本，共 148 段、44,400 帧（888 秒）。训练 120 段/36,000 帧，验证 28 段/8,400 帧，按 endpoint 划分，速度版本不跨集合；两边共享起始扶墙姿态。

Crocoddyl DDP 求每个端点；固定脚底方向的六维位姿 IK，关节限位保留 10% 裕量。每段为 anchor → target → anchor，关节采用 C2 五次插值，单程至少 2 秒或 3 秒，端点停留 0.4/0.6 秒。指令取每帧**实际 FK xyz**，不是强行令关节插值对应笛卡尔直线。最高关节速度 1.2 rad/s，加速度 2.5 rad/s²。

每帧检查：关节限位、移动腿与地面/墙间距、选定非相邻腿/身体凸包碰撞对、支撑脚与手接触不漂移，以及浮动基逆动力学平衡。接触 LP 包含单向法向力、内接摩擦锥（金字塔）、支撑脚 COP、接触扭矩和关节力矩余量。地面摩擦系数 0.7、墙 0.6；模型计算最大力矩为限值的 28.9%。

这是固定上身和脚底方向、选定采样范围内的**有限可达子集**，不是全机器人全部工作空间，也不是任意两个点之间轨迹的可行性证明。碰撞验证采用 USD 视觉凸包的选定对，不等同于 PhysX 全部碰撞对。逆动力学验证是理想接触模型下的可行性检查，不能替代训练后接触稳定性评估。

## 训练接口

任务：`cfg/task/G1/G1_wall_foot_reach.yaml`。

Student 的 `policy` 为 241 维：脚目标 xyz 3 + 角速度 3 + 重力方向 3 + 五帧关节位置 145 + 三帧动作 87。任务指令只有 xyz，不包含教师关节目标/未来运动/轨迹 ID。现有 PPO 的 `adapt_joint_module` 预测当前 29 维 `joint_target`，`adapt_module` 预测 privileged latent，student actor 使用两个预测结果；teacher 使用真实标签。预测值在原框架 VecNorm 归一化空间中训练，不应直接当弧度发送给电机。

全程保留固定世界坐标墙与支撑目标。关闭数据默认 +3.5 cm 高度偏移、关节目标裁剪和左右镜像增强；默认任务以外的原行为保持不变。新任务若误开镜像，PPO 会明确报错。

在已有 Isaac Lab 训练环境运行：

```bash
conda activate gentle
export PYTHONPATH=/home/xl521/IsaacLab/source/isaaclab:$PWD${PYTHONPATH:+:$PYTHONPATH}
# 默认本地训练，不上传 WandB；需要记录时设置 WANDB_MODE=online。
# WandB project 默认为 wall-foot-reach；run 名默认为 v1。
# 后续 run 可设置 WANDB_RUN_NAME=v2；项目可通过 WANDB_PROJECT 修改。
bash scripts/train_wall_foot_reach.sh train
CHECKPOINT=/absolute/path/to/teacher.pt bash scripts/train_wall_foot_reach.sh adapt
CHECKPOINT=/absolute/path/to/adapt.pt bash scripts/train_wall_foot_reach.sh finetune
```

沿用项目的 train → adapt → finetune。这里提供的是数据和训练入口，**尚未得到收敛的 motion-tracking policy**。`smoke_*.pt` 仅用于集成测试，不是可用控制器。旧的其他任务 checkpoint 输入维度不同，不应当作此任务的完整续训 checkpoint。

评估验证集可加 `task.command.dataset.mem_paths=[wall_foot_reach/val]`。部署/交互时，通过 `env.command_manager.set_target_foot_pos_b(xyz)` 给 student 当前 root 系的目标；支持 `[3]` 或 `[num_envs,3]`，`None` 恢复数据集目标。该接口只覆盖任务指令，训练奖励/教师标签仍来自数据集，适用于 student 推理，不用于带覆盖指令的 teacher 训练。外部调用方应限制目标和变化速度在验证范围内。

## 复现与验证

```bash
# Crocoddyl 环境，依赖版本见 requirements-lock.txt；输出目录必须尚未生成 manifest。
.venv-reference/bin/python scripts/references/generate_foot_reach_dataset.py
# 使用训练环境；目标 memmap 目录必须不存在，避免覆盖。
python scripts/data_process/pack_foot_reach_dataset.py
python scripts/data_process/verify_foot_reach_dataset.py
.venv-reference/bin/python scripts/references/preview_foot_reach.py

# 有限 4-env / 16-step rollout + 一次更新，禁用编译以快速检查。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/references/smoke_wall_foot_reach.py --phase train
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/references/smoke_wall_foot_reach.py --phase adapt --checkpoint "$PWD/artifacts/foot_reach/smoke_train.pt"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python scripts/references/smoke_wall_foot_reach.py --phase finetune --checkpoint "$PWD/artifacts/foot_reach/smoke_adapt.pt"
```

生成器支持 `--grid`、`--sobol`、`--seconds`、`--seed`、`--max-points`、`--output`，扩密时使用新输出目录，检查新 manifest 后重新打包。不要使用通用数据预处理脚本逐帧把最低脚归零，否则会破坏固定接触几何。

本次验证结果：逐帧源数据/训练数据 FK 一致，train/adapt/finetune 各完成 4 环境 × 16 步 rollout 和一次参数更新，checkpoint 依次加载成功；student 无教师标签输入的推理通过。初始化脚底位置最大误差约 3.69e-7 m。详细数值见 `artifacts/foot_reach/smoke_{train,adapt,finetune}.json`。这些检查仅验证接口与数值链路，不是训练后成功率评估。

## v2 左脚脚底外力扰动

训练配置 `command.foot_force` 默认启用。使用 high-level `hl/force/net_pull_ee.yaml` 的四阶段线性时序：休息 20–200 控制步 → 升力 25–100 步 → 保持 20–200 步 → 降力 25–100 步；50 Hz 下分别为 0.4–4、0.5–2、0.4–4、0.5–2 秒。每周期随机世界系三维方向、均匀采样 0–20 N 峰值，10% 周期峰值为零；episode 重置时清零。

每个物理子步在左脚实际脚底中心施力，只有 `left_ankle_roll_link` 有非零外力，PhysX 自动计算作用点到质心的力臂效果。student 仍为 241 维；teacher 的 priv 增加实际外力 root 系 xyz 三维。训练从头开始，三阶段使用新 schema 的 checkpoint，不能直接用 v1 teacher encoder 续训。WandB 记录 `force/foot_mean_N`、`force/foot_max_N`。评估入口默认关闭扰动；旧 checkpoint 的配置没有该字段时也不施加外力。

八卡三阶段默认预算启动：

```bash
WANDB_RUN_NAME=v2 bash scripts/train_wall_foot_reach_pipeline.sh
```

测试：`python -m unittest discover -s tests -p test_foot_force.py`；Isaac 集成短测添加 `--foot-force`，用缩短周期覆盖升力/保持/降力/归零，并验证左脚世界系施力点、力矩缓冲和仅左脚受扰动。
