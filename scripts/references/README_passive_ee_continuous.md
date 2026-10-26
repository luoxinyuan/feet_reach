# EE continuous motion tracking 修正版

旧版将参考直接跳到transition终点并将速度清零。本版使用独立命令 `PassiveEEContinuousCommand`，不修改旧实验。

## 参考与引导分离

- 参考帧每20ms仅前进一帧，保留AMASS关节/root速度以及中间姿态。
- 每隔25–50帧选择未来终点；该终点仅设置私有弹簧anchor，不修改参考时间、奖励目标或物理状态。
- 仅双手施力，每手范数上限30N；沿用40–160N/m拟合刚度与8N·s/m阻尼，切换anchor时0.2秒渐变。
- teacher阶段 `guidance_enabled=false`，物理施力及公开力观测严格为零；adaptation和student finetune启用引导。

## 数据与奖励

重用AMASS筛选后的256条训练/64条验证运动片段，不再重做Crocoddyl。另从每条片段的低关节速度帧生成静止hold片段，姿态保持不变、速度为零。训练采样80%运动、20%静止；hold片段不施力。因此不会随机撤力却继续要求student跟随移动参考。

数据位于 `dataset/passive_ee_continuous/{train,val}/{move,hold}`，由 `scripts/data_process/prepare_passive_ee_continuous.py` 生成。源clip划分沿用旧AMASS版本。

保留dense关节/root/关键点tracking，新增独立 `ee_tracking`（权重2，距离sigma为0.1/0.05m）。该项比较机体坐标系双手位置。验证报告的EE误差则用当前时刻参考手位置的世界坐标误差，**不与未来anchor比较**。

student输入仍只有本体历史、历史动作、实际双手力，无目标joint、anchor或phase。teacher可见特权参考。

## 训练与验证门槛

启动：`bash scripts/train_passive_ee_continuous_pipeline.sh`，默认GPU4–7，每卡12288环境。teacher从头训练1.2B环境步（原先400M只有254次rollout/PPO迭代，本版约762次）；adaptation100M、student400M。预算加大并不代表保证收敛。

teacher结束后，自动在256个验证初始化上分别测试连续动作和静止hold，冻结normalizer、确定性推理。通过以下实验验收阈值才进入student：

- 连续动作和hold的episode完成率均≥95%。
- 连续动作和hold的平均手部位置误差均≤8cm。
- 连续动作的平均关节MAE≤0.15rad。

未达标则pipeline非零退出，保存teacher checkpoint及 `teacher_validation/quality_gate.json`，不继续蒸馏，也不把固定预算跑完视为合格。验证按首次episode统计，连续片段最长3秒；hold也按其片段时长统计，不等于10秒或长时稳定性验证。

运行目录为 `outputs/passive-ee-continuous/`；`launcher.json` 和 `latest_pipeline.txt` 记录启动及本次目录。旧实验数据、checkpoint均保留。

## 实现检查

`scripts/references/smoke_passive_ee_continuous.py` 检查：切换anchor不改变参考时间/机器人姿态；每步只推进一帧；参考位置/速度与数据一致且运动速度非零；teacher零力；student仅手部受力且限幅30N；hold零力；student无目标输入；三个阶段有限梯度和checkpoint交接。

`artifacts/passive_ee_continuous/` 保存短测结果。`eval_probe` 使用未训练的短测checkpoint仅验证评估路径与失败门槛，不能作为训练效果。

## 本次启动

2026-10-21 12:18（America/New_York）启动，PID 3250701，目录 `outputs/passive-ee-continuous/20261021_121817`。已确认四卡实际PPO更新、teacher施力为零，启动吞吐约27万环境步/s，各卡显存约38GB；三个阶段短测通过。以上为运行验证，不代表训练效果。
