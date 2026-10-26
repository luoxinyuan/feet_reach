# EE dense tracking reward 修正版

新增独立任务 `G1/G1_passive_ee_spring`：使用 `guidance_mode=next_frame`，每控制步显式读取参考t+1帧（不是future_steps的第二项t+2），每物理步计算80 N/m纯弹簧力，每手范数限幅30N。hold强制零力；不采样transition、不拟合刚度、不加阻尼或切换ramp。dense tracking仍使用当前参考帧。当前运行实验未切换；通用pipeline的teacher开关会覆盖任务默认值，带力teacher需TEACHER_GUIDANCE=true。已检查下一帧索引、末帧截断、力方向/大小、限幅、hold和关闭引导时零力及仅双手施力；尚未进行该模式的实际仿真训练验证。

本版保留RootStudentPPO架构、目标仅teacher可见的输入划分、现有关节动作映射以及AMASS连续参考。没有复制CEER的EE/root interface。旧实验与checkpoint保留；新模型从头训练。

## 奖励修改

`PassiveEEDenseCommand`继承连续参考/引导逻辑，覆盖dense tracking奖励。

对关节误差或各关键点距离分别计算多尺度 `exp(-|error|/sigma)`，再平均；不再先将所有关节/关键点误差平均后计算指数。所有目标均来自当前参考帧，不使用未来引导anchor作为tracking目标。宽尺度保留较大误差时的学习信号，窄尺度强调精度；这是明确的reward改版，不保证必然改善训练。

| 项目 | 权重 | sigma |
|---|---:|---|
| Root世界位置 | 1.5 | 0.15/0.05 m |
| Root朝向 | 1 | 0.5/0.2 rad |
| 29关节位置 | 3 | 0.4/0.1 rad |
| 29关节速度 | 0.5 | 2.0/0.5 rad/s |
| 全身世界坐标关键点 | 2 | 0.2/0.05 m |
| 双手机体相对位置 | 2 | 0.15/0.05 m |
| 双手世界位置（新增） | 2 | 0.15/0.05 m |

关节速度直接使用仿真实际joint_vel与参考joint_vel，避免依赖reward函数调用次数维护差分历史。存活权重5→1，action_rate惩罚0.2→0.05；其他安全/正则项沿用。世界坐标手误差与机体相对手误差分别记录，不能混用或简单相减。

## 保留行为与独立修复

- 逐帧AMASS参考，保留姿态与速度；未来25–50帧终点仅生成引导力。
- teacher无外力；adaptation/student启用双手引导，每手30N上限、0.2秒过渡。
- 80%运动、20%静止hold片段，hold零力；数据复用passive_ee_continuous。
- 使用已修复的跨卡KL平均学习率调度，避免各rank学习率不同导致参数分叉。

## 训练

任务 `cfg/task/G1/G1_passive_ee_dense.yaml`，启动 `bash scripts/train_passive_ee_dense_pipeline.sh`。GPU4–7，每卡12288环境；teacher1.2B → adaptation100M → student400M环境步，W&B offline，每50次迭代存checkpoint。

teacher先通过验证门槛再启动student：连续与hold完成率≥95%，世界坐标手部误差≤8cm，连续动作关节MAE≤0.15rad。门槛报告明确标注世界坐标，另附局部手误差/root误差；未通过则保存报告并停止。验证是首次短片段，不能证明长时稳定性。

输出 `outputs/passive-ee-dense/`；launcher.json/latest_pipeline.txt记录实际运行。

## 验证

- tests/test_dense_tracking.py：目标处奖励为1、误差增大分数下降、每个关节梯度指向目标、batch独立和顺序不变。
- tests/test_root_student_distributed_lr.py：不同rank KL下学习率一致，DDP更新后参数一致。
- scripts/references/smoke_passive_ee_continuous.py --task G1/G1_passive_ee_dense：实际仿真关节索引/速度匹配、初始手部坐标匹配、连续参考、阶段交接、梯度有限、teacher零力和student限幅/目标隔离。
- artifacts/passive_ee_dense/保存日志；ddp_smoke为2卡3次PPO更新，模拟调度进度0.5以覆盖KL同步分支，不是恢复旧模型。

## 本次启动记录

### 带引导力对照组（2026-10-21）

按用户要求新增 `bash scripts/train_passive_ee_dense_guided_pipeline.sh`，GPU 0–3，每卡12288环境，从头训练；teacher、adaptation和student均开启同一双手引导力，每手上限30N，hold仍为零力。其余dense tracking奖励、网络、数据与1.2B/100M/400M预算保持一致。teacher验证沿用训练checkpoint的引导力设置，先检查带引导的连续动作和零力hold，达标才继续student。

本组输出独立保存到 `outputs/passive-ee-dense-guided/`，13:57:48 America/New_York启动，launcher PID 3307679。GPU 4–7的无外力teacher组继续运行。通用脚本默认仍为无外力teacher，新增入口显式设置 `TEACHER_GUIDANCE=true`。

2026-10-21 13:45:42 America/New_York 从头启动，launcher PID 3299785。运行目录 `outputs/passive-ee-dense/20261021_134542`，日志 `outputs/passive-ee-dense/launch.log`。上述数值测试、DDP回归、三个阶段仿真以及双卡PPO短训练均通过。正式四卡训练已验证到9,437,184环境步，PPO指标有限，teacher最大外力0N；速度约259,000环境步/秒，每卡显存约37,850MiB。按启动速度估计teacher阶段约75–85分钟，尚不能判断收敛效果；最终以独立验证门槛为准。

带引导组启动验证：实际配置guidance_enabled=true，已完成7,864,320环境步，PPO loss/梯度有限，teacher最大力30.000006N（浮点误差），速度约268,000环境步/秒。运行目录 `outputs/passive-ee-dense-guided/20261021_135748`。
