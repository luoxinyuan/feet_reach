# AMASS EE passive compliance 重训

本轮针对旧 EE student 的稳定性问题，将 Crocoddyl 参考动作替换为本地 AMASS 的 G1 重定向动作，并把**每只手的力上限改为 30 N**。从头训练，不加载旧实验权重。数据更换能否改善结果需要后续验证，尚不能断定旧数据是失败根因。

## 数据

`scripts/data_process/prepare_passive_ee_amass.py` 从 `/home/xl521/hiearchical_compliance/dataset/amass` 筛选站立手臂活动片段：256 条训练、64 条验证，每条150帧（按50 Hz，3秒），共48,000帧。每个源 clip 最多选一个窗口，train/val 按源 clip ID 分离；原始 AMASS 文件名不可用，不能保证原始采集文件互斥。

筛选要求：root 水平偏移不超过6cm、yaw变化不超过0.15rad、直立；脚部位置偏移不超过3.5cm、下肢关节变化较小；至少一只手活动超过6cm。保留原始重定向姿态，不做 IK、逆动力学或接触力优化。浮点数据转为float32、四元数归一化、初始root XY归零，并沿用本地AMASS数据的3.54cm足底高度修正。

数据：`dataset/passive_ee_amass/{train,val}`；筛选阈值、源clip ID及窗口起点均记录在 `manifest.json`。这些是运动学筛选，不是双脚严格固定或平衡可行性的证明。

## Transition 和施力

沿用原 EE 实验：同一片段内，从当前参考帧前后100帧、间隔5帧的候选中采样邻近目标；手部位移课程从2cm扩大到8cm。目标达到并短暂稳定，或2秒超时后切换，不重置机器人。约20%显式hold，以及无可用邻居的情况均撤除外力。

仅在两只手施加 `F = clip_norm(k*(anchor - x_hand) - 8*v_hand, 30 N)`，`k`仍按原方法在40–160 N/m范围拟合。力为三维向量，30N是每只手的范数上限，不是每轴上限；不额外施加力矩。为便于比较，未同时更改刚度、阻尼、奖励和蒸馏算法。

## 训练

任务：`cfg/task/G1/G1_passive_ee_amass.yaml`。启动：`bash scripts/train_passive_ee_amass_pipeline.sh`。

默认GPU 4–7，每卡12,288环境，合计49,152；teacher 400M → adaptation 100M → student finetune 400M环境步。每50次更新存checkpoint，阶段自动衔接，W&B offline。运行单独保存在 `outputs/passive-ee-amass/`；旧Crocoddyl及locomotion实验保留。

student仅观察本体状态历史、历史动作和实际双手施力，不接收joint target、目标位置或transition phase。奖励与原实验一致，使用teacher所用参考关节/全身目标。

## 验证与后续比较

`artifacts/passive_ee_amass/` 保存数据检查、transition邻居覆盖率和三个训练阶段的smoke报告。检查包括30N限幅、仅手部施力、hold零力、student目标隔离、有限梯度及阶段checkpoint交接。短测通过不代表策略已学会站立或柔顺。

本次两个力学单元测试及train/adapt/finetune三个集成短测均通过。训练/验证集中，2cm范围内有邻居的参考帧分别为66.6%/70.7%，8cm范围内为96.2%/95.9%；早期实际hold比例会高于20%。全部数据有限，train/val源clip ID无交集。

训练完成后应在AMASS验证集比较teacher/adaptation/finetune，重点看零力站立、物理跌倒比例和EE跟踪误差；不应把训练完成或episode完成率直接当作成功。

评估入口支持 `scripts/eval_passive_policies.py --task compliance --run-dir <本次运行目录> --controller student --scenario guided --output <评估目录>`，从checkpoint配置自动选择AMASS验证数据。零力站立用 `--scenario hold`，教师用 `--controller teacher`，蒸馏后用 `--controller adapt`。

本次启动于2026-10-20 23:41（America/New_York），launcher PID `2954494`；进程信息见 `outputs/passive-ee-amass/launcher.json`，运行目录见 `latest_pipeline.txt`。

已确认 `outputs/passive-ee-amass/20261020_234121` 开始实际四卡PPO更新。启动实测约27万环境步/s，每卡约41GB显存，施力峰值符合30N限幅（float32舍入误差约几微牛）。完整学习效果需训练后评估。
