# CEER residual teacher：离线力提示 + 实际物理施力

这是 `G1_ceer_offline_force_prompt_teacher` 的单变量对照：`physical_guidance=true`，让同一份离线力既进入 actor，也进入物理仿真。随机初始化 residual，冻结相同 CEER；不从失败的 residual checkpoint 续训。

## 保持不变

- AMASS/EE/静止混合集 + 下蹲集，权重 0.8/0.2；完整缓存 16,021 clips。
- 37 个候选施力点，当前帧 111D GT force，无 force history，无 estimator。
- actor 只看本体状态、自己的 reference 和当前力，不看下一帧目标；critic/reward 保留 motion target。
- 原 reference 累积、2 rad/s 速度上限、0.6 rad 偏差限制、reward、PPO、底层 checkpoint 均保留。
- 底层 `ceer_force_limit=5` 也保持与无物理施力实验一致，便于单变量比较。此前定位到的条件30/5失配及底层 dense tracking 瓶颈未在本实验中同时修正。
- 4B 总环境步，每卡20,480环境，3卡合计61,440，W&B online。

## 物理施力

缓存 `train37_budget60` 在控制帧 t 提供世界系 `F_i[t]`，同一帧先被 actor 观测，然后在本控制间隔的4个物理子步施加。使用实际 link 姿态把局部偏移变换到世界系；多个点属于同一 link 时，在 link 原点施加等效合力和偏移力矩：

`F_body = sum_i F_i`，`T_body = sum_i (R_body offset_i) × F_i`。

通过已有 PhysX world-frame force dispatch 提交，避免丢掉手/脚偏移点的力矩，也避免同一力被提交两次。每个物理子步重新计算力臂方向。这里的力矩仅用于等效表示偏移点的力，不是新增的独立旋转 prompt。

逐点力大小之和上限60 N，单点上限30 N；连续有效transition的单点变化率上限150 N/s，末帧为零，reset不保证跨片段连续。全缓存平均总力约5.389 N，实际引导较轻。施力点包括虚拟5cm偏移，并不要求落在机器人碰撞表面上。

力仍由参考轨迹离线计算，不根据实际偏差重新求解；因此不能保证自动纠偏或一定提高 tracking，本次实验正是检验物理引导的作用。

## 验证和启动

`smoke_offline_force_prompt.py --physical` 在真实物理仿真/PPO更新中验证：cache/frame对齐、力预算、actor目标隔离、逐子步力与力矩等效、非零物理buffer、底层冻结、actor更新、DDP精确同步、checkpoint往返。另运行原来的无物理力分支回归。

```bash
CUDA_VISIBLE_DEVICES=2,3,4 NUM_ENVS_PER_GPU=20480 \
  bash scripts/train_ceer_offline_force_physical_teacher.sh
```

入口先验证完整缓存，再保存独立代码快照、配置和底层hash。检查结果位于 `artifacts/offline_force_physical_smoke/`。正式运行路径和 W&B 链接见后续运行记录。

## 正式运行：2026-11-06

目录：`outputs/20261106_013202-ceer-offline-force-physical-teacher`。物理GPU 2/3/4，每卡20,480，总计61,440环境，4B预算。

W&B：https://wandb.ai/luoxinyuan-duke-university/ceer-force-prompt-teacher/runs/buc9a7i1

启动验证：已完成 13762560 frames，正式日志 physical_force_max_N 约30 N（非零），loss有限；W&B远端状态running。三卡显存约37–38 GiB；早期吞吐约20万frames/s，粗估全程5–6小时。原始无物理施力分支回归与三卡完整规模验证均通过。
