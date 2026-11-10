# CEER residual teacher：仅观测离线力，不实际施力

本实验验证：当前本体状态＋自身累积参考＋当前力提示，能否替代下一帧motion target来训练residual motion tracking。没有force history，没有force estimator，也没有物理引导力。

## 固定结构

- 冻结此前下蹲增强的CEER low-level `outputs/20261101-ceer-lowlevel-crouch-student-gt-joint-target/113246/finetune/wandb/run-20261101_114716-r2e0ta3u/files/checkpoint_4500.pt`，绕过joint predictor，直接输入residual累积后的29维joint reference。
- residual仍为29维，50 Hz；`q_ref += .02 * 2 * tanh(action)`，保持已有软关节限位和0.6 rad参考偏差限制。reset参考为机器人实际q。
- actor：原有本体状态/关节与动作历史、自己生成的reference及实际跟踪差、**当前37点×3维GT力提示**。不输入motion ID、时间、下一帧q或reference motion；保留proprioception历史不等于force history。
- 111维力提示按固定点顺序排列，从世界系转换到实际robot root系，除以30 N；使用独立`force_gt_`字段，不进行VecNorm。没有额外观测参考Jacobian或未来施力点坐标。
- critic仍能看目标和privileged状态；奖励仍为原dense tracking和稳定性奖励。延续最近no-EE版本，额外EE位置tracking项关闭，保留joint/root/keypoint等奖励。
- PPO沿用512/256网络、原超参数、同步KL学习率调度；只删除本次不使用的force estimator和蒸馏teacher副本。
- 物理外力/力矩缓冲区始终为零，`force_apply_world=False`，关闭PhysX外力调用。力提示不经过物理slew滤波；缓存已经离线平滑，避免额外延迟或reset时提示缺失。

## 全量缓存

训练源：原`20261027-ee-root3-multi10/dataset`的15,695个clip（4,102,328帧）和`crouch_lowlevel/train`的326个clip（80,852帧）；抽样权重0.8/0.2。包含此前EE/静止等增强，沿用原采样分布，不只训练离线评测的51个窗口。

为避免两个数据源的motion ID冲突，将选中clip打包到新的只读用途数据视图，重编号ID/step，其他状态数据保持原值；不更改原始数据。每个clip有独立cache offset，不跨片段求Δq。

力定义保持离线基线：37个固定候选点，`tau_prompt=2.5969108014710645*Δq`，`A=[J_i,q^T]`。SOCP先最小化`||AF/K-Δq||²`，再允许至多1e-4 rad额外RMS误差并最小化总用力＋时间平滑。每帧`Σ||Fi||₂≤60N`、每点≤30N、每点变化率≤150N/s。不包含root合力/合力矩匹配。

全量预计算按64个transition分块；块首约束相对上一块末力的变化率，平滑目标也包括块间差分，防止块边界跳变。与短窗口基线相比，新增了跨块连续性约束。片段第一帧自由，片段末帧因没有下一步提示为零；episode结束不要求跨clip连续。只约束参考序列连续性，不约束随机reset之间的力提示。

当前编码只覆盖驱动关节增量，不单独编码root平移/朝向增量。37点包含末端5cm虚拟偏移；本轮不实际施力，不涉及这些点的物理接触面验证。

## 运行

```bash
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/offline_force_baseline/prepare_training.py \
  --output artifacts/offline_force_baseline/train37_budget60 --workers 160
CUDA_VISIBLE_DEVICES=2,3,4 NUM_ENVS_PER_GPU=20480 \
  bash scripts/train_ceer_offline_force_prompt_teacher.sh
```

缓存只有全部成功、有限性检查通过后才写`manifest.json`；任何失败均阻止训练入口启动。训练默认4B总环境步，W&B online，项目`ceer-force-prompt-teacher`。源数据、施力点、增益、预算、几何hash和索引记录在缓存manifest；每clip保存求解状态与误差。

## 检查

`scripts/references/smoke_offline_force_prompt.py`：三卡真实Isaac两次PPO更新，验证actor输入隔离、111维无force history、提示frame索引（含reset）、力预算、物理外力为零、actor更新、底层冻结、DDP参数一致、checkpoint往返。小环境通过结果在`artifacts/offline_force_baseline/prompt_smoke_3gpu/`。

`tests/test_offline_force_baseline.py`：反向力不能逃避总预算、时间平滑、块边界单帧slew、偏移点Jacobian有限差分，4项通过。

容量测试在`artifacts/offline_force_baseline/prompt_bench_20480/`，额外预留全量force cache等量显存。正式启动与状态另见运行目录。

## 本轮额外诊断

全量源数据的约2.1%（主集）/2.4%（下蹲集）关节增量分量超过现有residual单步0.04 rad上限。本轮按“其他结构保持一致”保留该上限，没有为改善结果额外改action speed。后续评估需区分force编码误差、reference生成误差和冻结底层跟踪误差；日志分别记录reference与实际joint的MAE。

正式启动前执行`validate_training_cache.py`：逐clip检查有限性、总力/单点力/跨块变化率、ID与step索引，并对100个随机clip逐字段比对源数据。训练入口保存独立代码快照与源码hash，防止训练中途共享工作区编辑影响本次实现。

## 正式运行（2026-11-05）

已启动：物理GPU 2/3/4，每卡20,480环境，总计61,440环境，4B预算（2034个完整PPO batch）。运行目录：`outputs/20261105-ceer-offline-force-prompt-teacher/`。完整缓存16,021段、4,167,159个有效transition，平均总力5.389 N，离线Δq重建相对误差5.995%，RMSE 0.0007844 rad。

W&B：https://wandb.ai/luoxinyuan-duke-university/ceer-force-prompt-teacher/runs/h9yq8vnk

确认前8次更新完成，15,728,640环境步，物理引导力日志持续为0，loss/梯度有限。三卡显存约41 GB/卡；初期总吞吐约18万–20万frames/s，据此粗估全程约5.6–6.2小时，实际以稳定后吞吐为准。W&B远端状态已核实为running。

入口端到端在线测试（仅3072步，不是正式训练）：https://wandb.ai/luoxinyuan-duke-university/ceer-force-prompt-teacher/runs/cfcxedpy 。完整数据源拷贝/索引和力约束检查见`artifacts/offline_force_baseline/train37_budget60/validation.json`。
