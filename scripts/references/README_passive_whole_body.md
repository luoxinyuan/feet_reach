# 全身 passive compliance：LIMMT AMASS full

## 数据与训练

本地数据实际目录为 `/home/xl521/hiearchical_compliance/dataset/limmt_no_foot_gentle_amass_full`，不是未经筛选的全部原始AMASS。共8082条存储片段、4509578帧，全部浮点字段通过有限值检查。保留每段251–1000帧（按当前50Hz参考约5–20秒），不额外截成3秒。

`scripts/data_process/prepare_passive_whole_body.py`生成只读数据视图，共7692训练片段/4286066帧、390验证片段/223512帧。按存储clip ID固定hash分组；没有原始采集文件映射，不能保证原始录制级互斥。全部片段参与相应分区，训练每次reset重新抽clip，连续参考逐帧推进，换施力点不reset。源数据脚踝高度基准为0，沿用0.03540915m足底修正。不额外做IK或动力学可行性过滤。

采用CEER PPOPolicy：teacher直接输入joint_target、student通过本体历史及实际力预测latent/joint目标。没有EE/root target接口注入student。全关节dense tracking使用MotionTrackingCommand原生奖励，保留平衡/足部正则；机器人、动作缩放0.5及无延迟沿用之前统一实验，关闭对称增广。

`/home/xl521/hiearchical_compliance/cfg/train.yaml`中CEER默认total_frames=4B。新pipeline teacher4B，adapt100M（沿用本实验蒸馏预算），student finetune4B；不能称100M来自CEER原配置。teacher先验证完成率≥95%、全身关节MAE≤0.15rad、世界关键点误差≤0.15m，未通过则停止并保存报告。该全身关键点门槛替代旧EE手部门槛；此数据无独立frozen hold分区，改为guided与zero运动对照，不冒称测试了静止保持。

## 施力方案

9个候选点：torso及左右shoulder_roll、elbow、wrist_roll、hand_mimic原点；均为上半身。每次随机1–3个点，按参考部位速度加权选点，权重有非零下限。1–3秒后先将旧力降到零，再换点，始终最多3个点。没有直接torque，没有脚底施力。

每控制步显式读取下一帧body position作为anchor；局部意图为 `k*(anchor-current)+4*(reference_velocity-actual_velocity)`，k每次选点在60–100N/m采样。root合力意图为位置误差×60加线速度误差×8，限制40N；yaw力矩意图为角误差×8加角速度差×1.5，限制4Nm。后者只用来分配接触力，实际torque buffer始终零。

9未知数的批量正则最小二乘：局部力误差权重1、与上一实际力差权重1、合力及yaw残差权重4；yaw以0.3m特征力臂归一化。单/双点布局合力残差>5N+50%目标或yaw残差>0.5Nm+50%目标时增加为3点。并非每个布局都可实现完整目标，因此记录残差。

求解后投影到合力/yaw不反向的半空间；若有限次投影仍不满足，则该次目标力退为零。之后统一缩放，限制每点30N、所有力范数之和60N，保留合力方向。每个物理步对力做共同插值，保证各点变化率≤150N/s；这不是简单每轴截断。实际力存在有限跟随延迟，快速意图变化时可能短暂方向相反；日志单独记录实际反向比例，不能把目标方向检查等同于实际始终精确满足。局部动作项是软约束，未保证所有局部方向与全局意图同时可满足。

物理引擎在当前body原点施加力，由真实力臂产生力矩。本方法是几何意图分配，不使用质量惯量、逆动力学或足底接触力解算，也不保证外力独立实现下一姿态。

student公开观测仅本体历史、历史动作、9个固定部位上的实际机体坐标系力；没有anchor、目标姿态、参考速度或目标wrench输入。稀疏力不能唯一编码所有全身动作，这仍是需要验证的训练假设。

## 验证与运行

`tests/test_body_guidance.py`验证平移/转向符号、退化布局、inactive点、单点/总量/变化率约束。`smoke_passive_whole_body.py`验证实际仿真、最多3点、torque零、t+1参考、换点变化率、student目标隔离，以及teacher/adapt/finetune的有限梯度。双卡3次PPO更新覆盖DDP KL同步。`artifacts/passive_whole_body/`保存结果。

入口 `bash scripts/train_passive_whole_body_pipeline.sh`；输出 `outputs/passive-whole-body/`，每50次迭代保存checkpoint。自动guided/zero验证会记录逐关节误差、全身关键点误差、合力与yaw残差和接触点数。短片段验证不能证明任意长交互能力。

## 本次正式启动

2026-10-21 23:14 America/New_York，GPU4–7，每卡12288环境，PID3669629，运行目录`/home/xl521/feet_reach/outputs/passive-whole-body/20261021_231413`。已完成多轮四卡PPO，约17.5万环境步/s，每卡总显存约25–26GiB（含4/5卡已有小任务），GPU计算利用率100%；teacher4B阶段按启动速度约6–7小时，后续以实际吞吐为准。没有终止其他GPU任务。三阶段smoke、数据全量有限值检查、双卡PPO、whole-body验证脚本均已通过。尚未验证训练收敛或大范围柔顺能力。
