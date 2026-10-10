# G1 Wall Foot Reach

G1 扶墙脚部 reaching：左手扶墙、右脚支撑、左脚跟踪实时 root 坐标系下的笛卡尔目标。包含参考轨迹生成、teacher / adapt / finetune 训练、固定目标精度评估和网页遥操作。

## Proactive EE tracking（PointNet）

新增双脚支撑、左手 EE tracking 实验：student 输入实时 root 坐标系下的 EE target xyz、本体历史和 PointNet 编码的局部点云，保留 target joint estimator。墙位置独立随机采样，25% 场景无墙；通过 tracking、接触力上升率和持续力惩罚学习接近行为，不使用距离触发的减速规则。

- Crocoddyl 数据集：80 条连续轨迹、18,798 帧，按目标点划分为 56 条训练、24 条验证轨迹。
- 已通过三阶段 PPO、checkpoint 衔接、student 推理、PointNet 梯度及物理接触反馈的集成检查；尚不代表已验证学会主动减速。
- 八卡训练，每卡默认 4,096 个环境（共 32,768）；teacher / adapt / finetune 分别为 40 亿 / 10 亿 / 20 亿环境步。

```bash
bash scripts/train_proactive_ee_pipeline.sh
```

W&B 项目默认 `wall-foot-reach`，run 名为 `proactive-ee-v1`（后续附加 `-adapt`、`-finetune`）；输出位于 `outputs/proactive-ee/`。当前使用带噪声和缺测的仿真表面点云，尚未接入真实 depth / SLAM。

数据生成、验证及输入约定见 [Proactive EE 说明](scripts/references/README_proactive_ee.md)。数据集和 checkpoint 仍需另行生成或复制，不随 Git 上传。

## 运行环境

训练和评估需要 Isaac Lab / Isaac Sim、PyTorch，以及本项目依赖。在已配置好的仿真环境中，从项目根目录运行：

```bash
export PYTHON=/path/to/isaaclab-environment/bin/python
export ISAACLAB_PATH=/path/to/IsaacLab
"$PYTHON" -m pip install -e .
"$PYTHON" -m pip install wandb
"$PYTHON" -m wandb login
```

已有 W&B 登录凭据或 `WANDB_API_KEY` 时无需重新登录。还需准备 `artifacts/foot_reach/workspace.npz`、`artifacts/foot_reach/manifest.json`，以及 checkpoint 配置引用的轨迹数据和机器人资产。W&B 下载入口只下载权重。

## W&B checkpoint 评估

直接把 W&B / CoreWeave Forge run 链接传给 `--checkpoint`，程序下载后加载 policy 和观测归一化参数：

```bash
bash scripts/eval_wall_foot_reach.sh \
  --checkpoint 'https://forge.coreweave.com/wandb/luoxinyuan-duke-university/wall-foot-reach/runs/wall-foot-reach-finetune-20261004_164441' \
  --video
```

默认从完整可行点集中，从初始脚位置出发，每次选择距离上一个目标最近的未选点，连续执行 20 个不同目标（不再随机采样，也不限于 validation 点），开始时重置并稳定 1 秒，之后连续切换目标，中途不重置。直接下发目标，留出 2 秒 reaching 时间，统计最后 1 秒窗口的平均位置误差；成功/失败只根据是否稳定判定；失稳即停止，剩余目标标记为未尝试。输出指标和录像到 `artifacts/foot_reach_eval/<时间戳>/`。

## W&B checkpoint 网页控制

```bash
bash scripts/eval_wall_foot_reach.sh \
  --checkpoint 'https://forge.coreweave.com/wandb/luoxinyuan-duke-university/wall-foot-reach/runs/wall-foot-reach-finetune-20261004_164441' \
  --web --port 8765
```

等待终端出现 `WEB_READY`，在 VS Code Remote SSH 的 **Ports / 端口** 面板转发 `8765`，然后打开 `http://localhost:8765`。网页提供 X/Y/Z 目标调整、暂停和重置；加 `--video` 可同时录像。`Ctrl+C` 停止。

两个入口共用 checkpoint 下载逻辑：优先选 final，否则选编号最大的 checkpoint。加 `--wandb-file checkpoint_7500.pt` 可指定 run Files 中的文件。缓存位于 `.cache/wandb-checkpoints/`。也支持 `run:entity/project/run-id` 和本地文件路径。

## 数据与训练

- [参考姿态生成](scripts/references/README.md)
- [轨迹数据集、三阶段训练与外力配置](scripts/references/FOOT_REACH.md)
- [评估参数、指标和网页使用说明](scripts/foot_reach/README.md)
- 任务配置：`cfg/task/G1/G1_wall_foot_reach.yaml`
- 八卡三阶段训练：`WANDB_RUN_NAME=v2 bash scripts/train_wall_foot_reach_pipeline.sh`

`dataset/`、`artifacts/`、`outputs/`、W&B 日志及训练 checkpoint 不在版本控制中。新机器需要先生成或复制评估数据，再运行上述入口。

旧版项目说明保留在 [README_legacy.md](README_legacy.md)，供历史配置和原框架使用方式参考。
