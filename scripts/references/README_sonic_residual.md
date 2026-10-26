# Frozen SONIC + force-prompt residual student

实现日期：2026-10-22。训练代码已经经过真实 Isaac Lab / GPU 验证；尚未进行长时间收敛训练，因此不能把接口与 PPO 更新验证当作已经学会 compliance 的证据。

## 方案与输入隔离

```text
AMASS 下一帧 ──→ 上半身 1–3 个引导力 ──→ 仿真机器人
      │                         └──→ student 的测得外力观测
      ├──→ dense tracking reward
      └──→ privileged critic（仅训练）

机器人自身状态 + 测得外力 + 自己的参考/误差/参考更新速度
                         ↓
                 residual student（29维）
                         ↓
               积分更新自己的关节参考
                         ↓
        frozen SONIC G1 encoder → FSQ → decoder
                         ↓
              SONIC joint action → PD → 机器人
```

- student actor 只有 `policy` 一个输入；公开观测白名单为角速度、重力、关节位置历史、关节速度、前次 residual、9 个候选上半身受力点的实际力，以及自生成的关节参考、误差、更新速度。
- critic 可见 `priv`、`joint_target`，但其输出不作为 actor 输入。没有 teacher/adapt/finetune 三阶段，也不加载旧 student。
- AMASS 使用现有 `passive_whole_body_full/train`（7692 个片段，源为本地 LIMMT full AMASS）。训练 reset 仍按现有流程随机采样实际机器人初态；随后 reference 从**实际关节状态**初始化。episode 内绝不从 AMASS 重置或推进 SONIC reference。
- 引导力、dense joint/root/keypoint tracking reward 延用 `PassiveWholeBodyCommand`。只施加力，不直接施加 torque：每个点最多 30 N，总力幅值预算 60 N，力变化速率 150 N/s。
- `student_train: false` 是继承的旧 reward 配置字段，用于保留 dense tracking；本任务的优化对象始终是 residual actor/critic，没有可训练 SONIC 或 teacher。

## Reference 与 SONIC 接口

默认控制频率 50 Hz。网络输出无界 `a_t ∈ R^29`，其物理含义为参考更新速度：

```text
requested_delta = 0.02 × 2.0 rad/s × tanh(a_t)
q_ref_next = q_ref + bounded_delta
```

更新受软关节限位及相对实际关节 ±0.6 rad 的跟踪误差边界约束。只限制本次增量，不把已保持的参考强行拉回实际关节；零 residual 精确保留参考，避免外力一推，目标自动跟随的行为。参数位于 `task.action.reference_speed` 和 `max_reference_error`。

SONIC 的 10 帧 G1 窗口全部重复刚更新的 `q_ref_next`，窗口内部速度设为零；这里是 receding setpoint / 零阶保持，每个控制周期会重新更新。公开的 `reference.velocity` 记录实际参考增量除以 dt，不能与窗口内部的零速度混淆。该窗口没有任何 AMASS 未来帧。

原版 SONIC 还要求 root orientation 输入。本版本采用**实际 heading 下的直立姿态**作为固定规则：不给 AMASS 的目标 yaw，也不固定世界 yaw。学生只输出 29 维关节 residual，没有额外 root residual。转向由力、身体响应和关节参考共同产生。这个接口没有独立指定 root roll/pitch/yaw 的自由度，不能据此保证复现 full AMASS 中所有躺卧、翻滚等运动；与原直接输出动作的 student 也不是严格等价。

SONIC 原生权重被冻结、`eval()`、`no_grad()`，不在任何优化器内。每张 GPU 一份模型，环境批量前向；reference、10 帧本体感觉历史、上一条低层 action 均独立按环境维护/reset。动作通过官方 default pose + action scale 转为 PD target，29 个关节全部按名称映射。PD 增益、armature、力矩/速度限制均取官方 MODEL_12 定义；机器人几何使用项目现有 full-collision G1 USD，并开启 self-collision。并非移植整套官方仿真场景。

注意原版官方 `command_multi_future` 的排列是全部 q 帧后接全部 qd 帧，再 reshape；本实现保留该排列，并通过官方 ONNX 数值对照验证。

## 官方资源与文件

- [官方源码](https://github.com/NVlabs/GR00T-WholeBodyControl)，固定 commit `45c03facbf848e0c743091b7d9c17952096ff908`。
- [官方训练权重](https://huggingface.co/nvidia/GEAR-SONIC/tree/bfd764a/sonic_release)，原版 `sonic_release/last.pt`，revision `bfd764a`。
- `.cache/sonic/manifest.json` 记录原始权重、tensor-only actor 和配置 SHA-256；加载时校验 actor/config，其他 variant 必须重新适配验证。
- 使用官方 `UniversalTokenModule` 和 `BaseModule`，只实例化 G1 encoder 与 dynamics decoder，不使用 SMPL/teleop encoder 和训练辅助 decoder。
- 原 checkpoint 含旧 trainer 元数据。准备脚本通过 `weights_only=True` 和明确白名单把元数据映射为无行为占位对象，提取 tensor state dict；不需要安装旧版整个 trainer。

代码入口：

| 文件 | 用途 |
|---|---|
| `active_adaptation/utils/sonic.py` | 冻结官方模型、reference、历史缓存 |
| `active_adaptation/assets/sonic.py` | SONIC 机器人参数 |
| `active_adaptation/envs/mdp/sonic_residual.py` | residual → SONIC → PD |
| `active_adaptation/envs/mdp/commands/sonic_compliance.py` | 原有引导力/reward，公开观测白名单 |
| `active_adaptation/learning/hierarchical/sonic_residual_ppo.py` | public-only actor、DDP KL 同步、优化器保存恢复 |
| `cfg/task/G1/G1_sonic_compliance.yaml` | 全身引导训练任务 |
| `cfg/algo/sonic_residual.yaml` | PPO 配置 |

## 启动与验证

在本机器使用 gentle Python。准备脚本把附加模型依赖安装到 `.cache/sonic/deps`，不升级 Isaac Lab 环境里的 torch/numpy：

```bash
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/sonic/prepare.py --with-onnx
/home/xl521/software/miniconda3/envs/gentle/bin/python scripts/sonic/check_native.py
```

四卡训练入口（每卡 2048 环境，共 8192；总帧数跨所有 GPU 统计）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_ENVS=2048 TOTAL_FRAMES=4000000000 \
  bash scripts/train_sonic_residual.sh
```

默认总帧数参照现有 CEER/whole-body 的 4B 设置，不表示已经启动长训练。`NUM_ENVS` 是每卡环境数，`CUDA_VISIBLE_DEVICES` 决定 worker 数。支持 `PYTHON_BIN`、`ISAACLAB_ROOT`、`MEMPATH`、`OUTPUT_DIR` 环境变量；尾部参数透传 Hydra。checkpoint 保存 residual actor、critic、优化器、观测归一化和配置，SONIC 由单独的固定资源加载。

```bash
export PYTHONPATH=/home/xl521/IsaacLab/source/isaaclab:$PWD
export MEMPATH=$PWD/dataset
export OMP_NUM_THREADS=4
python_bin=/home/xl521/software/miniconda3/envs/gentle/bin/python
CUDA_VISIBLE_DEVICES=0,1,2,3 "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/sonic/smoke_train.py --num-envs 2048
CUDA_VISIBLE_DEVICES=4 "$python_bin" scripts/sonic/check_standing.py
```

## 实测记录

机器当前 GPU 上，float32 原生 SONIC 与官方 ONNX 最大输出差 `1.07e-6`；批量与逐环境推理最大差 `1.31e-6`。2048 / 4096 batch 平均前向耗时约 4.10 / 7.42 ms（模型推理，不含仿真/PPO）。

四卡 × 2048 环境，32 步 rollout，2 次 PPO 更新，每次 2 epochs × 2 minibatches：

- 各卡使用不同随机种子；参数更新后逐项完全一致，学习率一致。
- actor 参数确实更新；SONIC 参数/缓冲区逐项不变，没有梯度且不属于优化器。
- 毒化真实 motion 的关节/root 目标不影响公开观测；修改私有 target 输入不改变 actor 输出。
- reset 一个环境不会影响其他环境的 reference/history。
- 最终初始学习率 `3e-5`，更新 KL 约 `0.0048`、`0.0046`，全部 loss/梯度指标有限。
- PyTorch peak allocated 约 4.0 GB/卡；该数值不包含全部 PhysX/renderer 显存，不应直接据此估算最大环境数。

实际训练资产上，8 个环境、零外力、零 residual 连续站立 10 秒，禁用自动 reset：最低 root 高度 0.752 m，最大倾斜 0.0120 rad，最大水平漂移 0.0225 m。随后通过 residual 更新一个肘关节的正弦参考 6 秒，平均跟踪误差 0.0275 rad，保持站立。这验证非零 residual 确实能改变 SONIC 的跟踪行为，并非训练效果评估。

详细结果位于 `artifacts/sonic_residual/{native,standing,smoke_4gpu_rank0}.json`；正式 launcher 短跑另存于 `artifacts/sonic_residual/launcher_4gpu`。这些都是验证产物，不能用来判断 force compliance 已收敛。下一步长训练后应评估 guided / zero-force / force-release、未见片段，以及站立初始化的手动交互。

正式 launcher 已通过四卡短跑、checkpoint 保存及仅主进程评估；该 checkpoint 在单卡恢复训练成功，Adam step 从 40 增至 80，优化器状态并未丢失。汇总证据：`artifacts/sonic_residual/validation.json`。
