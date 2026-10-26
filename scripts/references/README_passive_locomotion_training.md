# Passive whole-body locomotion 最小训练验证

本实验使用 AMASS 的 G1 重定向动作作为教师参考，与 EE passive compliance 共用 teacher → adaptation → student finetune 架构。引导规则只使用位置和朝向误差，不做逆动力学、接触力求解或轨迹可行性优化。

## 数据与 transition

- 来源：本地 `/home/xl521/hiearchical_compliance/dataset/amass`，按 50 Hz 读取。
- `scripts/data_process/prepare_passive_locomotion.py` 筛选直立、速度适中的片段，并从低速且双脚高度接近的静止片段生成 hold 数据。
- `dataset/passive_locomotion_simple`：训练 256 条运动、14 条站立；验证 64 条运动、5 条站立。划分按本地 clip ID，不能保证原始 AMASS 源动作互斥。
- 训练按 80% 运动、20% 站立采样，参考动作连续播放以保留步态；每隔 25–50 帧选择新的未来 root 位置、yaw 作为私有 anchor。切换 anchor 不重置机器人。到片段末尾或失败才重置。

## 简单引导规则

在 pelvis 施加水平力 `F_xy = clip_norm(60 * (anchor_xy - actual_root_xy), 30 N)`，竖直分量为零。在 torso_link 施加绕世界竖直轴的力矩 `tau_z = clip(8 * wrap(anchor_yaw - actual_yaw), ±4 Nm)`。每次 anchor 切换从上一时刻的力/力矩用 0.2 s 线性混合到新引导值。站立样本的外力、外力矩为零。

这里力矩直接模拟人对躯干的转向引导；不要求它与平移力满足动力学方程。训练仍保留常规步态奖励（例如脚滑惩罚），它们不参与引导力生成。

## 教师与学生

- 教师：特权参考动作观测，dense root、关节和关键点 tracking reward。
- 学生：本体观测、关节历史、历史动作、机体坐标系下的实际施加 wrench；没有 joint target、anchor、目标速度或 phase 输入。
- adaptation 蒸馏教师 latent；finetune 使用相同的 dense 参考动作奖励。关节监督来自教师使用的参考轨迹，并非另一个并行 teacher 机器人实际执行后的关节角。
- 当前最小验证把仿真中的施加 wrench 作为已知输入；真实部署需要相应的力感知或估计。

## 运行与验证

任务配置：`cfg/task/G1/G1_passive_locomotion.yaml`；引导实现：`active_adaptation/envs/mdp/commands/passive_locomotion.py`。

启动：`bash scripts/train_passive_locomotion_pipeline.sh`。默认 GPU 0–3，每卡 12,288 环境，合计 49,152；教师 400M、adaptation 100M、student finetune 400M 总环境步，三阶段自动交接 checkpoint，每 50 次迭代存盘。W&B 使用 offline 模式。GPU 4–7 的 EE 实验独立运行。

日志位于 `outputs/passive-locomotion/`，`latest_pipeline.txt` 指向本次运行目录，`launcher.json` 记录后台进程。以上是固定步数预算，不代表已达到收敛。

2026-10-20 18:24（America/New_York）已启动：`outputs/passive-locomotion/20261020_182424`，launcher PID `2813843`。首轮实际 PPO 更新约 312k 环境步/s，GPU 0–3 各占用约 38 GB 显存；这是启动实测值，后续吞吐可能变化。

单元测试 `tests/test_simple_locomotion_guidance.py` 覆盖限幅、方向、yaw wrap 和零力。集成测试 `scripts/references/smoke_passive_locomotion.py` 覆盖真实仿真施力位置、三个训练阶段梯度和 checkpoint 交接、student 目标隔离及 transition 不重置。结果保存在 `artifacts/passive_locomotion_simple/smoke_{train,adapt,finetune}.json`。

本次两个单元测试及 train、adapt、finetune 三个集成测试均已通过。

早先 `README_passive_locomotion.md` 描述的是离线动力学探索，当前训练不使用该方案。
