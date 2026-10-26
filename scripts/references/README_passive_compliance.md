# Passive whole-body compliance：最小验证

G1、双脚站立、29 关节控制。仅在左/右 `hand_mimic` 施加外力，每条参考轨迹移动一只手，另一只手及下肢保持参考站姿。这里 passive 指 **没有运动目标输入、由外力引导**，不表示已证明控制理论上的无源性。

## 数据

`generate_passive_compliance_dataset.py` 从项目 USD 构造 Pinocchio 模型，用 Crocoddyl DDP 求解手部姿态，接五次插值轨迹；不是完整的全身动力学轨迹优化。左右手分别对 384 个 Halton xyz 点求解，固定各自初始手部朝向。采样盒为 x=[.10,.55]、|y|=[.05,.50]、z=[.65,1.30] m。

每帧检查关节限位、固定双脚、所选手臂—躯干/对侧手臂碰撞对，以及摩擦、足底力矩和关节力矩约束下的接触逆动力学 LP。不是全连杆碰撞认证。

最终 208 条、48,454 帧（50 Hz）；训练 140 条、验证 68 条，按 endpoint 分割，两种速度不会跨 split。左手 75 个 endpoint、右手 29 个；接受点范围见 `artifacts/passive_compliance/coverage.json`。这覆盖的是固定手朝向下的可行子空间，未声称完整工作空间或任意全身形态覆盖。

## Transition 与外力

借用 [Whole-Body Dexterity §III-B](https://arxiv.org/html/2610.22184#S3.SS2) 的可行状态初始化、邻近目标课程、hold 和连续切换思想；不复现其物体抓持 reward、grasp-mode graph 或 diffusion distillation。

- 从 clip 随机帧初始化真实机器人；在同一 clip 的 ±100 帧中、以 5 帧间隔寻找候选。邻居 EE 距离 1 cm 到随机半径，半径上限从 2 cm 增至 8 cm；20% hold。无邻居时退化为 hold。
- 后一帧的 whole-body pose 是本次静态 dense reference。关节误差 <.08 rad、双手位置误差 <2 cm、平均关节速度 <.2 rad/s 连续 10 步，并至少经历 .5 s 后切换；2 s 超时也切换。切换目标不重置机器人，episode 重置才重新采样初始状态。
- 从两帧之间抽 12 个参考点，固定 b=8 Ns/m、虚拟 m=1 kg，用带正则的有界最小二乘拟合每手 k∈[40,160] N/m：`min Σ||k(x_goal-x)-b*v-m*a||² + .01(k-80)²`。这是 Cartesian surrogate，不能保证任意 joint displacement 由 EE 力唯一实现。
- 每个 200 Hz 物理步施加 `F=clip_norm(k*(anchor-x_EE)-b*v_EE, 60 N)`；anchor 固定在该环境世界坐标。力作用点为实际手部位置，显式外加 torque 为零，其他身体部位无外力。hold 完全撤除外力。

## Teacher / student

采用已有 `RootStudentPPOPolicy`，teacher 输入本体/力观测和 privileged reference（包括全部 29 关节）。Teacher 使用 dense root、joint、keypoint tracking 加 standing/regularization reward。

Student **唯一输入组是 `policy`**：角速度、重力方向、5 帧关节位置、3 帧历史 action、实际施加的双手 root-frame 力。无目标 joint、EE target、anchor、刚度、transition phase 或目标误差输入；无 joint target estimator。Privileged critic 可见 reference，仅参与训练。

训练顺序：teacher PPO + latent estimator → student rollout 下 latent 蒸馏 → student dense-reward PPO。当前 student 的 joint reward 对齐 **teacher 使用的同一个 reference**，不是另外运行一个 teacher 机器人并用其实际关节轨迹做标签。最小实现避免了额外 teacher 仿真实例；若要研究 teacher 实际执行轨迹的影响，应另采集 teacher rollout 标签。

## 运行

```bash
.venv-reference/bin/python scripts/references/generate_passive_compliance_dataset.py --side left
.venv-reference/bin/python scripts/references/generate_passive_compliance_dataset.py --side right
python scripts/references/merge_passive_compliance_dataset.py
# 使用装有 torch / tensordict 的训练 Python：
$PYTHON scripts/data_process/pack_foot_reach_dataset.py --source artifacts/passive_compliance --output dataset/passive_compliance
bash scripts/train_passive_compliance_pipeline.sh
```

默认 GPU **4,5,6,7**，每卡 12288 环境（首次启动为 2048，后经吞吐测试调整）；teacher/adapt/student 分别 4 亿/1 亿/4 亿步。W&B 默认 offline，日志和 checkpoint 留在本地。可通过 `NUM_ENVS`、`TEACHER_FRAMES`、`ADAPT_FRAMES`、`STUDENT_FRAMES`、`WANDB_MODE` 覆盖。运行路径记录在 `outputs/passive-compliance/latest_pipeline.txt`。

验证入口：`tests/test_passive_compliance.py` 检查拟合目标、力方向、阻尼、限幅和 hold；`scripts/references/smoke_passive_compliance.py` 检查仿真 rollout、仅手部受力、无目标 student 推理、privileged 输入扰动不改变 student 输出、PPO 更新、阶段 checkpoint 衔接及无重置 transition。报告位于 `artifacts/passive_compliance/smoke_*.json`。通过这些检查不代表已经学会顺应；需在验证 split 上评估跟随误差、稳定性及撤力后的行为。

## 本次启动（2026-10-20，美东时间）

单卡 train → adapt → finetune 集成检查全部通过，student 推理输入仅 `policy`，三阶段更新指标有限；力学单元测试 2 项通过。已确认后台四卡进入 PPO 更新，观察到 1,572,864 个环境步，rollout 约 12 万步/s，GPU 4–7 各约 10.4 GB 显存，GPU 0–3 空闲。启动 PID 记录于 `outputs/passive-compliance/launcher.json`，本次目录为 `outputs/passive-compliance/20261020_171846/`。长训练是否收敛需看后续 checkpoint 和验证结果。

## 环境数调优（2026-10-20）

四卡、同一 checkpoint、相同 PPO 参数，逐一测试 4096/8192/12288 环境/卡；每次 12 次更新，丢弃前 3 次，取完整 rollout+PPO 更新吞吐中位数。短测不等同于收敛速度保证，大 batch 会减少固定总步数下的更新次数。

| 每卡环境数 | 四卡完整吞吐 | PyTorch 峰值 allocated/卡 |
|---|---:|---:|
| 4096 | 187,728 步/s | 11.29 GB |
| 8192 | 243,787 步/s | 22.56 GB |
| 12288 | 341,122 步/s | 33.82 GB |

选择 **12288/卡，共 49152**；nvidia-smi 总显存约 41/49 GB/卡，四卡利用率可达 100%。原 2048/卡长期完整吞吐约 10–11 万步/s。原训练停止在 checkpoint 900 后，从其已完成的 236,191,744 步恢复；保留模型、归一化和课程进度，teacher 只训练剩余约 1.64 亿步。旧 checkpoint 无 Adam 状态，优化器重建，学习率按 checkpoint 恢复。总预算仍为 teacher 4 亿 + adapt 1 亿 + student 4 亿，未增加预算。

新运行路径见 `outputs/passive-compliance/latest_pipeline.txt`，PID 与恢复来源见 `launcher.json`，测试结果见 `outputs/passive-compliance/benchmark/results.json`。每 50 次更新保存 checkpoint。入口支持 `RESUME_CHECKPOINT` 和 `TEACHER_DONE_FRAMES`；训练日志新增 `iteration_fps`（包含 PPO 更新）、累计 `env_frames` 和 `cuda_peak_GB`。

恢复运行目录：`outputs/passive-compliance/20261020_175959/`。已确认超过 500 个控制步、跨完整 episode 重置继续运行，完整吞吐约 33–34 万步/s，总显存约 41 GB/卡，无 OOM。按既定步数预算预计美东时间 18:35–18:45 完成三阶段；这不是收敛保证。
