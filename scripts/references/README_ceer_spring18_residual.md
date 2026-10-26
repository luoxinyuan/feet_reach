# CEER residual：18 点逐帧弹簧引导

本实验按用户选择沿用原来的逐帧弹簧生成器，不使用 Jacobian 优化。冻结同一份
通用 CEER，绕过 joint predictor，重新训练 residual student。AMASS LIMMT full、
dense tracking 奖励、residual 积分和 PPO 参数保持原配置。

## 候选点和施力

共 18 个固定身体局部点，具体坐标见
`cfg/task/G1/G1_ceer_spring18_compliance.yaml`：

- 原来的 9 点：躯干原点、左右肩/肘/腕/手。
- 躯干前后左右 4 点：高度为躯干局部 z=0.18 m，前后 x=±0.08 m，左右 y=±0.10 m。
- 左右上臂中段 2 点、左右前臂中段 2 点，依据 USD 的关节间距设置。
- 头部仅 `head_mimic` 原点 1 个中心点，不设置头部前后左右多个点。

每次仍采样 1–3 点，保持 1–3 秒；每点 30 N，各点模长之和 60 N；变化率
150 N/s；换点前完全卸力。逐帧使用对应局部点的下一帧参考位置作为弹簧 anchor。
参考速度由相邻参考点位置差计算；实际点速度使用连杆原点速度加 omega×offset。
原实现阻尼用了 body COM 速度；本实现改用实际施力点速度，使偏置点运动一致。

多个点可以属于同一连杆。仿真 API 每连杆只有一个 wrench，因此使用
`F_link=sum(F_point)`、`M_link=sum((p_point-p_link)×F_point)`，在连杆原点施加。
这里 moment 只是点力的等效表示，不是额外生成的旋转控制。相反方向的两个力
合力为零时，其力偶仍会保留，不能覆盖或丢失。

student 外力输入按固定 18 点顺序排列，从 27 维增至 54 维；总 public observation
从 381 维增至 408 维。输入是平滑后的真实点力，先按点记录，再转为 root 坐标。
同一连杆的不同点不会被合并成一个 observation。没有新增未来目标输入。
旧 9 点 residual checkpoint 不兼容 408 维输入；frozen CEER 本身不变。

## 验证

- 6 个纯张量测试：同连杆力偶、总 wrench/机械功等价、限幅、平滑、方向约束。
- 三卡各 256 环境、2 次真实 PPO 更新：CEER 权重冻结、actor 更新、DDP 同步、
  target poisoning、公有输入隔离、reset 隔离、18 点/408 维输入检查。
- 在仿真中检验同连杆前后反向力的力偶；使用 PhysX COM Jacobian 独立核对
  18 个偏置点的线速度（1e-4 容差）。Jacobian 仅用于测试，不用于生成力。
- 报告位于 `artifacts/ceer_spring18_residual/`。训练中记录每个点实际受力的比例。

## 启动与 W&B

```bash
CUDA_VISIBLE_DEVICES=5,6,7 NUM_ENVS=16384 WANDB_MODE=online \
  bash scripts/train_ceer_spring18_residual.sh
```

默认 40 亿帧，3 卡共 49,152 环境、每批 32 步、2,543 次 PPO 更新，实际
3,999,793,152 帧。与原 CEER 基线规模一致。正式 run 使用独立 frozen CEER 副本。

用户要求后续训练上传 W&B：训练启动脚本默认已改为 online，仍支持显式
`WANDB_MODE` 覆盖。单元测试和 smoke 检查可 disabled；正式训练不应默默退回
离线模式。当前已运行的历史离线任务不会因此自动切换或重新启动。

## 本次正式训练（2026-10-23）

- GPU 物理编号 5、6、7，每卡 16,384 环境。启动时间 13:35 EDT。
- 目录：`outputs/ceer-spring18-residual/full_20261023_133334`，后台 PID 1227685。
- W&B：https://wandb.ai/luoxinyuan-duke-university/ceer-spring18-residual-compliance/runs/z0len5ib（online）。
- 8 次更新吞吐测试：稳定中位数 131294 帧/秒，
  每卡约 31–32 GiB 总显存；预计 8–9 小时，约 10 月 23 日 22:00 EDT /
  北京时间 10 月 24 日 10:00 完成，受共享机器负载影响。
- `frozen_ceer.pt` 为原 checkpoint 的独立副本，SHA256
  `637ff52b2712b4ad3c5fa4b8004a85ff11567e703a92532345867de28c45dc7c`。
- `source_snapshot/`、`source_hashes.json`、`.hydra/` 和 `launch.json` 保存复现信息。
- 日志 `train.log`；每 50 次更新保存 checkpoint 到 `wandb/*/files/`。
- 基线是原 9 点逐帧弹簧 CEER residual；正在运行的 Jacobian 实验保持原配置。
