# CEER residual：离线分段 Jacobian 引导

2026-10-24。冻结原始 `.cache/ceer/checkpoint_final.pt`，绕过 joint predictor；student 的 29 维 residual 累积更新自己的关节参考。沿用 CEER residual PPO、dense tracking reward、AMASS LIMMT full 数据和 4B frame 预算。actor 只读取本体状态、实际外力、自身参考及误差，没有真实 motion target。18 点 force observation 使 public observation 为 408 维。

## 力生成

使用训练机器人同一个 USD 导出的关节几何，离线计算世界坐标点 Jacobian。关节名义 torque 为 `80*(q[t+1]-q[t])`，保留旧 Jacobian 的 reference delta 项；离线不加入依赖实际状态的 feedback 项。根部前馈为 `60*delta_position + 8*reference_velocity`（位置差分速度），yaw 为 `8*delta_yaw + 1.5*yaw_velocity`（旋转增量差分）。没有逆动力学、重力补偿或足底接触求解，也没有新增松手保持奖励。

拟合 root 六维广义力、waist 和双臂；腿部 torque 不作为上半身点力的重建目标。根部平移/旋转、腰部、左右臂分别归一化，静止手臂的非零重建 torque 同样计入误差。运动仍由 frozen CEER 执行；离线重建好不等于实际闭环 tracking 一定好。

18 候选点沿用扩展 spring 布局，含头部中心一个点。先在 10 Hz 的参考帧筛选单点、双点和由优良双点扩展的三点组合；在 0.2 s 网格上动态规划，综合拟合误差、点数时间成本和点集合切换成本。不是全组合全局最优求解。合成左→右→左测试自动产生三个段，双臂测试自动选择两侧点。

固定候选分段后，通过 OSQP 在完整 50 Hz 轨迹上联合求所有帧的力；平滑按物理点身份计算，非按可换序的 slot。每点 30 N、总和 60 N、变化率 150 N/s；当前使用保守 L1 力球与分量 slew 约束，保证欧氏范数上限，不代表充分使用了整个欧氏可行域。每个点集合切换前一个参考帧所有力归零，保证物理子步插值时也不会出现超过三点的交接。这会造成短暂引导减弱，是本版明确的折中。

最终检查平滑后每段整体及左右臂相对误差，均值门槛 0.85。相邻合格段合并，保留至少 1 s 的连续区间。不是每帧误差都小于 0.85；门槛是经验参数。暂不自动修复失败段，失败段有记录并排除，不能将筛选后训练表现与旧全数据指标直接等同。训练采样这些连续区间，力切换不重置机器人。完整原始 clip 的力缓存也保留，方便在完整 val 集公平评测。

第一版每条轨迹保存一个确定性优化方案，训练随机采样轨迹/起始帧，没有再独立随机重抽施力点。支持后续增加多套近优缓存。

## 数据与验证

训练集原始 7,692 clips / 4,286,066 frames，接受 3,260,660 frames（76.08%），合并为 9,161 个连续区间。验证集原始 390 clips / 223,512 frames，接受 163,026 frames（72.94%）。训练继承原有单 episode 最多 1,000 帧设置。

同样 32 条训练 clip 的 pilot：9 点接受率 82.37%、平均相对误差 0.68290；18 点接受率 85.19%、误差 0.67811。保留 18 点；这只是离线依据，不是 policy 效果提升证明。

- `tests/test_segmented_guidance.py`：左右左分段、双臂支持、静止零力、切换零力和约束。
- `scripts/references/validate_segmented_cache.py`：逐 clip 检查所有帧、力上限、总预算、slew 和切换。
- `scripts/sonic/smoke_train.py --backend ceer --segmented`：真实 Isaac PPO、target isolation、冻结权重、更新和 DDP 同步；新增离线 USD FK/Jacobian 与 PhysX 对照。
- train/val cache 的 `manifest.json` 和 `validation.json` 含逐 clip 质量、采用点数和覆盖统计。

## 入口

```bash
# 需要先导出几何，具体 Isaac 环境变量见旧训练说明
python scripts/references/export_guidance_geometry.py
python scripts/data_process/prepare_segmented_guidance.py --out artifacts/ceer_segmented_jacobian/train18 --workers 24
python scripts/data_process/prepare_segmented_guidance.py --split val --out artifacts/ceer_segmented_jacobian/val18 --workers 8
CUDA_VISIBLE_DEVICES=5,6,7 NUM_ENVS=16384 WANDB_MODE=online bash scripts/train_ceer_segmented_residual.sh
```

正式训练信息见同目录 `artifacts/ceer_segmented_jacobian/training_status.json`（实际路径位于项目根 artifacts）。W&B project 为 `ceer-segmented-jacobian-residual-compliance`。

## 本次正式运行

- 输出：`outputs/ceer-segmented-jacobian-residual/full_20261024_122951`
- 物理 GPU：5、6、7；每卡 16,384 环境，共 49,152。
- 预算：3,999,793,152 实际 frames / 2,543 updates；32 rollout steps，5 PPO epochs，8 minibatches。
- W&B：https://wandb.ai/luoxinyuan-duke-university/ceer-segmented-jacobian-residual-compliance/runs/00lci6cu
- 启动后已从 W&B 远端读回 11,010,048 frames，actor/critic loss 有限；约 231k frames/s，初步预计约 5 小时。显存约 31 GB/卡。
- checkpoint 每 50 updates 保存并上传，首次在 update 50；启动核验时尚未到首次保存点。
- `source_snapshot/`、`source_hashes.json` 和冻结底层 checkpoint 副本保存在运行目录。首次启动的 Hydra 相对路径错误日志保留为 `.startup_path_error`；修正为相对项目根解析缓存后正式运行正常。
- 原始缓存全部 4,286,066 帧通过有限值、点数、力预算及 slew 验证；最大实际缓存单点力 14.26 N、总预算 27.32 N，30 N 是上限而非强制幅值。

## 筛选缓存续训（2026-10-26）

使用 `train18_single_hand_ee_static_screened_20261025` 训练同一 18 点 segmented Jacobian residual，4B frame，GPU 2–7 共六卡，每卡 16,384 env。W&B run：
[screened-20261026-6gpu](https://wandb.ai/luoxinyuan-duke-university/ceer-segmented-jacobian-residual-compliance/runs/kthwynfv)。
本次运行目录为 `outputs/ceer-segmented-jacobian-residual/screened_20261026`，tmux 会话 `ceer-segmented-screened`；进度写入该目录的 `training_status.json` 和 `train.log`。筛选仅从原采样索引移除一个超速片段，具体阈值与统计见筛选缓存的 `SCREENING_REPORT.md`。

## 多力方案缓存（2026-10-26）

新训练缓存为 `artifacts/ceer_segmented_jacobian/train18_single_hand_ee_static_screened_multi10_20261026`，保留原方案并扩增为每条动态轨迹 10 套力方案：7,692 条原始动作和 4,209 条单手 EE 轨迹均为 10 套；3,000 条静止 hold 仍为单一零力方案。原始动作的新增方案采用平滑时间幅值变化，并逐条通过训练片段的双臂拟合门槛和力约束；单手轨迹使用不同的平滑脉冲。少数拟合敏感轨迹只能使用较小幅度的方案变化。总计 122,010 个轨迹方案通过完整逐帧校验。

训练配置已切换到该缓存。每个环境在 episode reset 时，从该轨迹的 10 套有效方案中均匀随机选一套，并在 episode 内固定；静止 hold 选唯一的零力方案。`guidance/variant_index_mean` 可用于检查采样。校验最大单点力 20.00 N、总力 39.12 N、变化率 150.00 N/s，均不超过现有 30 N / 60 N / 150 N/s 上限；所有点切换前一帧力为零。审计结果见缓存目录 `validation.json`，方案来源和逐轨迹筛选结果见 `variant_manifest.json`。

multi10 训练已于 2026-10-26 启动：GPU 5–7，每卡 16,384 环境，共 49,152 环境，训练预算 4B frames。运行目录为 `outputs/ceer-segmented-jacobian-residual/multi10_20261026_163438`，tmux 会话 `ceer-segmented-multi10`；W&B：[multi10 run](https://wandb.ai/luoxinyuan-duke-university/ceer-segmented-jacobian-residual-compliance/runs/4vqahv90)。启动后首轮更新正常，状态见运行目录 `training_status.json`。
