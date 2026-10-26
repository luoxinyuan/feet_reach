# Dense tracking + AMP：受控对照

2026-10-24。本次不是 BFM-Zero 复现，也未引入 Forward–Backward 网络。保留上一版 frozen CEER residual、离线分段 Jacobian 18 点力缓存、相同训练数据视图、相同 dense reward/termination/PPO 设置，从头训练，只增加无条件运动先验奖励。没有新增松手保持模式。

## 训练设计

- Baseline：`outputs/ceer-segmented-jacobian-residual/full_20261024_122951`。
- AMP：`outputs/ceer-segmented-amp/full_20261024_173253`。
- 两者均 3 GPU × 16,384 env、32 rollout steps、5 PPO epochs、8 minibatches，3,999,793,152 frames / 2,543 updates。
- 使用完全相同的已通过力质量筛选的 9,161 个连续区间。专家只取该 train view，不使用 val。
- 预计算 100,000 对真实相邻帧，均匀采样区间、再均匀采样区间内合法起点；不会跨原始 clip 或筛选边界。
- 每帧 83 维：root 高度、重力方向、root 局部线/角速度、29 joints q/qd、双手/头/双脚相对 root 的局部位置。判别器输入两帧共 166 维，无 motion ID、参考目标、force、residual reference 或动作命令。
- 共用特征函数处理专家与实际机器人状态；专家 root 高度使用与训练相同的 z offset。fixed expert mean/std normalization，std 下限 0.1，归一化后裁剪 ±10。
- 判别器 MLP 256→128→1，ELU；LSGAN expert=+1、policy=-1；真实样本 input gradient penalty=10，Adam 1e-4，weight decay 1e-4，每 PPO update 做 4 次判别器更新，batch 4096。负样本一半当前 rollout，一半 65,536 容量历史缓存。
- 运动先验 `r_style=clip(1-0.25*(D-1)^2,0,1)`，权重从 0 到 2，前 100 updates 线性增加；与原奖励一样乘 0.02 s。最终最多增加 0.04/step。dense reward 原样保留。
- 本轮 rollout 的奖励使用判别器更新之前的参数计算；判别器随后更新，再进行 PPO。排除 is_init/done transition，不拼接 reset 前后状态。
- actor 仍只读取原来的 408 维 public observation。`amp_state_` 是独立训练字段，不进入 actor/critic/VecNorm。部署只需 actor 与 frozen CEER，判别器不参与控制。
- D 梯度显式 all-reduce；三卡验证参数完全一致。checkpoint 保存 D、归一化和 optimizer，resume 时策略历史样本池重新填充。

## 为什么先不同时改别的

第一版不加 force-conditioned discriminator，也不放宽累计 tracking error 终止条件，保证差异主要来自运动先验。它不保证恢复能力或外力语义自动改善。暂不加入其他训练技巧，否则不能辨认收益来源。只有一个训练 seed，结论是初步对照，不是统计显著性证明。

## 验证

- `tests/test_compliance_amp.py`：平面平移/heading 变换不改变运动特征，奖励有界。
- `scripts/sonic/smoke_train.py --backend ceer --amp`：真实三卡 PPO、D 更新与同步、权重冻结、target isolation、checkpoint state 往返。
- W&B 分别记录 dense reward、AMP reward/权重/贡献、真假判别准确率、gradient penalty 和 replay 数量，避免只看总 reward。
- 旧分段 policy 的评测见 `artifacts/ceer_segmented_eval_20261024`：完整 val（不再按训练质量门槛筛选）、512 个 matched first episodes、独立力/零力配对测试及完整视频。后续 AMP 用同样协议。

## 启动

```bash
PYTHONPATH=$PWD python scripts/data_process/prepare_compliance_amp.py
CUDA_VISIBLE_DEVICES=5,6,7 NUM_ENVS=16384 WANDB_MODE=online bash scripts/train_ceer_segmented_amp.sh
```

运行状态与 W&B 链接记录于 `artifacts/ceer_segmented_amp/training_status.json`。
