# Dense-link force prompt：CEER residual 三阶段训练

2026-10-27。按用户确认使用数据中的 **23 个关节 link**，排除 pelvis、head_mimic、left_hand_mimic、right_hand_mimic；手部提示位于 wrist_roll_link。是对刚体施加笛卡尔力，不是对关节添加扭矩。

## 离线力与 K

来源：`dataset/passive_whole_body_full/{train,val}`，原有 LIMMT/AMASS 全量视图。保持原来的 50 Hz 时间基准（来源 manifest 标记 fps_assumed=50），不改变数据采样率。train 4,286,066 个有效帧；val 223,512 帧。底层共享存储各有 4,509,578 行，未选中的行写零；绝不将 val 帧用于 train 校准。

每个 link 在统一世界系下使用

`F[t,i] = clip_norm(K * (x_ref[t+1,i] - x_ref[t,i]), 30 N)`。

只依赖参考相邻帧，不使用实际机器人位置、目标反馈、Jacobian、逆动力学或接触模型。各数据片段末帧为零，不跨 clip 求差。固定全局 K，不逐帧归一化。转动只通过不同 link 的位移产生力；纯原地姿态变化仍不能被位置差唯一表达。

预处理入口：

```bash
PYTHONPATH=$PWD python scripts/data_process/prepare_dense_link_guidance.py
```

输出：`artifacts/ceer_dense_link/full23/{train,val}/forces.npy`、`motion_offsets.npy`、`valid_next.npy` 和 manifest。各缓存约 1.2 GB（float32），包含原始 motion_id/step 的直接索引；manifest 保存数据视图及缓存文件 SHA256。已经生成的完整缓存拒绝覆盖，需要重做时指定新 `--output`。

K 的四个筛选上限均只由 train 决定：

- 单点最大 ≤30 N：K≤150.2004 N/m。
- 全部点力模长之和 P99≤60 N：K≤105.7093 N/m。
- 全部点力模长之和最大≤120 N：K≤49.3049 N/m。
- 合力最大≤80 N：K≤36.5104 N/m。

取最小值 **K=36.510354 N/m**。60/120/80 N 是明确的保守工程筛选预算，不能证明动态稳定或最优学习效果；不是每帧改变增益。没有把每个点都推到30 N。验证集只复核，未用于选择 K。

| 名义力指标 | train | val |
|---|---:|---:|
| 单点最大 | 7.29 N | 5.15 N |
| 力模长总和均值 | 5.73 N | 5.47 N |
| 力模长总和 P99 | 20.72 N | 20.53 N |
| 力模长总和最大 | 88.86 N | 31.94 N |
| 合力最大 | 80.00 N | 30.32 N |
| 关于参考 root 的合力矩最大 | 11.25 Nm | 8.15 Nm |

完整统计和候选 K 对比见 `artifacts/ceer_dense_link/CALIBRATION.md`、`full23/calibration.json`。这些是参考几何下的名义载荷，实际机器人偏离轨迹后力矩可能不同。

运行时延续150 N/s的每点变化率上界（所有点共用插值系数），每5 ms物理步更新、50 Hz residual 控制。GT标签使用**实际施加的力**，不是未限速的未来缓存力，因此 reset/力变化时也与物理一致。23点都保留独立力观测。没有加入随机删点、零力保持奖励或额外扶稳控制器。

## 三阶段接口

底层始终是原通用 CEER `.cache/ceer/checkpoint_final.pt`，权重及其状态 estimator 冻结，绕过 joint predictor。高层输出29维 residual，沿用已有参考积分、软关节限位和0.6 rad实际跟踪偏差约束。

1. **teacher**：public proprioception/自身生成参考 + 69维真实力（本体系，除以30 N）；PPO，原 dense tracking reward，原 termination。未来目标仅供 reward/privileged critic。默认4B frames。
2. **distill**：student仅用public状态和10帧、930维因果 proprioception/实际底层动作历史；估计23×3维力。学生自己的 rollout 状态上，冻结 teacher 使用 GT 力提供 residual mean；以 tanh 后 residual 的MSE监督student，以实际力MSE监督estimator。默认1B frames。
3. **finetune**：student仍只用估计力；原dense reward做PPO，附加0.05权重的teacher residual模仿损失，继续监督estimator。默认4B frames。

阶段2/3冻结阶段1的VecNorm；force/history字段采用固定物理尺度、不进VecNorm。阶段2从teacher初始化actor/critic，阶段3从蒸馏checkpoint初始化。跨阶段重置optimizer/update计数；同阶段可恢复optimizer。教师权重、学生、估计器、optimizer、normalizer均存checkpoint。

Force estimator沿用仓库CEER direct-privileged adaptation的512→256 MLP与MSE监督方式，扩为69维连续输出，初始化新输出头/网络，**不是直接加载旧单施力点force estimator权重**。这是独立于冻结CEER底层状态estimator的高层force estimator。

历史包含root角速度/重力、29关节位置与速度、最近实际底层动作；没有GT力、motion ID、下一帧目标、contact force或参考轨迹。reset后清空该环境旧历史并用当前测量填充，重复读取不会重复推进历史。teacher和student输入中的参考都来自自己的residual积分。

Student部署入口的in_keys严格为 `[estimator_history_, policy]`；teacher为`[force_gt_, policy]`。监督和critic可访问GT，但不进入student执行入口。估计器在本轮PPO完成后才更新，PPO使用rollout时保存的估计力，保持behavior distribution和importance ratio一致。

多点力未必能由本体状态唯一恢复；三阶段实现不代表已经验证force estimator精度或单点交互泛化。必须分别评估GT teacher、GT-force student和estimated-force student。

## 启动及阶段恢复

```bash
# 全流程；默认W&B online。以下是启动示例，本次尚未执行全量训练。
CUDA_VISIBLE_DEVICES=5,6,7 NUM_ENVS=2048 \
  bash scripts/train_ceer_dense_force_pipeline.sh

# 分别运行一个阶段。OUTPUT_DIR必须是未使用的新目录。
CUDA_VISIBLE_DEVICES=5,6,7 NUM_ENVS=2048 \
  bash scripts/train_ceer_dense_force_stage.sh teacher
CHECKPOINT=/absolute/path/to/teacher/checkpoint_final.pt \
  bash scripts/train_ceer_dense_force_stage.sh distill
CHECKPOINT=/absolute/path/to/distill/checkpoint_final.pt \
  bash scripts/train_ceer_dense_force_stage.sh finetune
```

可设置`PIPELINE_DIR`、`CACHE_ROOT`、`TEACHER_FRAMES`、`DISTILL_FRAMES`、`FINETUNE_FRAMES`；单阶段使用`OUTPUT_DIR`、`FORCE_CACHE`、`TOTAL_FRAMES`。恢复同阶段时指定该阶段checkpoint及新输出目录；训练帧数表示本次新增预算。默认环境数是保守启动值，未做大环境吞吐调优；默认总预算约9B，各阶段按rollout batch向下取整。

W&B项目：`ceer-dense-link-force-compliance`，每阶段独立run。launch前验证缓存，保留force manifest和frozen CEER hash。原有其他训练配置不修改。

## 验证

- `tests/test_dense_link_guidance.py`：clip边界、不跨片段、平移/旋转一致性、点力限幅与密集合力约束。
- `scripts/references/validate_dense_link_cache.py --cache ... --full-hash`：全量有限性/30 N限制、原始ID索引、片段末端、8192帧原公式重算及SHA256。
- `scripts/references/smoke_dense_force.py`：真实Isaac三阶段、三卡各16环境，两次更新；检查actor/estimator变化、frozen CEER与teacher不变、目标/GT隔离、DDP同步、checkpoint往返。结果在`artifacts/ceer_dense_link/smoke/`。
- 正式shell入口的短串联验证在`artifacts/ceer_dense_link/pipeline_smoke_v2`，每阶段仅1024frames、W&B online；这些checkpoint只用于程序验证，不代表训练效果。

正式入口验证的W&B runs（每阶段仅1024 frames）：

- Teacher: https://wandb.ai/luoxinyuan-duke-university/ceer-dense-link-force-compliance/runs/ux5h1zi0
- Distill: https://wandb.ai/luoxinyuan-duke-university/ceer-dense-link-force-compliance/runs/ahcdevgd
- Finetune: https://wandb.ai/luoxinyuan-duke-university/ceer-dense-link-force-compliance/runs/tg9e9s58

三者远端均为finished。跨阶段递归校验VecNorm所有嵌套张量完全一致，监督teacher权重也完全一致。验证汇总：`artifacts/ceer_dense_link/pipeline_smoke_v2/validation.json`。

## 全量启动：2026-10-27（America/New_York）

当前运行目录 `outputs/ceer-dense-force/full_20261027_230935`。物理GPU 5/6/7，每卡20,480环境，总计61,440；三阶段4B/1B/4B依次执行，W&B online，启用expandable_segments分配器。

该运行使用不可变代码快照 `outputs/ceer-dense-force/full_20261027_230652/code` 以及当前运行目录内的frozen_ceer.pt。保留该快照目录供后续自动阶段使用。

已对比16,384/20,480/24,576环境的微调吞吐。原24,576方案在整批teacher标签推理时OOM，已改为最多65,536行的分块推理；65,537行真实teacher权重一致性检查最大误差4.18e-7。优化后的24,576单卡连续5次更新通过，但实际三卡teacher训练仍超过显存，故退回20,480重新启动，失败尝试未保存训练checkpoint。不能用单卡短测通过替代实际并行训练验证。

单卡短测warm平均吞吐：16,384约51,690、20,480约57,753、分块24,576约60,592 frames/s。测试期间保留了其他用户的少量GPU进程，因此这些数字仅作为容量/吞吐参考。

详情：`artifacts/ceer_dense_link/benchmark/summary.json`；启动和健康状态：`artifacts/ceer_dense_link/training_status.json`。
