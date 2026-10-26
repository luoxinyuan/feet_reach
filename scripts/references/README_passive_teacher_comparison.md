# CEER tracking baseline 与逐帧弹簧 teacher 直接目标输入实验

2026-10-21 18:02 America/New_York 从头启动两组teacher，每组四卡、每卡12288环境、1.2B环境步。入口 `bash scripts/train_passive_teacher_comparison.sh ceer` 或 `spring-direct`。仅teacher训练，结束后自动运行256环境guided/hold验证和原质量门槛；不启动student。

## 前四卡：CEER PPO普通tracking baseline

任务G1_passive_ee_ceer，算法原生ppo_train/PPOPolicy，直接joint_target输入actor、256维priv latent、512/512/256 actor及critic。沿用ppo_train的学习率5e-4、初始噪声1.0、熵系数0.005→0.002。无引导外力，无EE/root interface提取模块。

关节位置、差分关节速度、世界关键点、机体相对EE奖励直接调用MotionTrackingCommand原生实现。普通tracking权重root位置/朝向0.5、线/角速度1、关节位置1、关节速度0.5、关键点1、EE2；使用原尺度joint_pos=[0.4,0.2]、joint_vel=[2,1]、keypoint=[0.3]、ee=[0.2]。存活5、动作变化惩罚0.2。

这是CEER原生PPO和普通dense reward在统一EE实验环境中的baseline，不是现有CEER finetune checkpoint的完整配置复刻：不包含其impedance奖励或root/wrist目标接口；保持同一g1_col_full机器人、全关节动作缩放0.5、无动作延迟、现有公开本体观测与priv观测。固定支撑命令声明不支持对称增广，故显式关闭。原生ppo_train参数也不等于checkpoint里finetune阶段参数。

## 后四卡：spring-direct

沿用逐帧弹簧teacher的全部数据、奖励、网络宽度和超参数，仅将joint_target从priv编码器输入移到已有direct_priv_keys通道，直接拼到teacher actor，避免29维参考关节必须经latent恢复。priv_in_keys=[priv]、direct_priv_keys=[joint_target]。已有直接目标预测器为未来student提供预测值，部署policy仍不读取真实joint_target；本轮未训练/验证student。

施力不变：80N/m、下一参考帧EE减实际EE、每手30N、hold零力。该改动检验目标编码瓶颈假设，不保证改善。与旧spring的比较只有目标输入路径这一项实验改动；与CEER组比较则同时存在算法、奖励和外力差异，不应作单因素归因。

## 数据、验证与输出

两组复用passive_ee_continuous，256训练/64验证运动片段，每段150帧；80%运动、20%hold。没有扩展工作空间或长时轨迹。本轮评估新增每个关节MAE，区分EE位置准确和手臂姿态准确。

两组均通过2GPU×128环境×3次PPO短训练，有限loss；CEER最大力0，spring最大力30N（float32误差）。短训练设置resume_frames=600M仅用于覆盖中期KL同步调度，不加载任何checkpoint。日志位于artifacts/passive_teacher_comparison。

正式输出分别为outputs/passive-ee-ceer、outputs/passive-ee-spring-direct，其launcher.json、latest_pipeline.txt与launch.log记录运行状态。每50次更新存checkpoint。
