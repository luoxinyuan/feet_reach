# 扶墙脚部固定点评估 / 网页遥操作

在项目根目录、已配置 Isaac Lab / PyTorch / wandb 的环境中运行。评估与网页模式共用 `--checkpoint`，支持 W&B / CoreWeave Forge run 链接，下载后加载 student policy 及观测归一化参数。

## 固定目标评估与录像

```bash
bash scripts/eval_wall_foot_reach.sh \
  --checkpoint 'https://forge.coreweave.com/wandb/luoxinyuan-duke-university/wall-foot-reach/runs/wall-foot-reach-finetune-20261004_164441' \
  --video
```

默认连续测试 20 个最近邻目标，中途不重置。可用 `--points 20` 修改数量，或用 `--targets /path/targets.json` 指定按顺序执行的目标列表。

## 网页遥操作

```bash
bash scripts/eval_wall_foot_reach.sh \
  --checkpoint 'https://forge.coreweave.com/wandb/luoxinyuan-duke-university/wall-foot-reach/runs/wall-foot-reach-finetune-20261004_164441' \
  --web --port 8765
```

加 `--video` 可同时录像。只接受包含 `cfg`、`vecnorm` 的 adapt / finetune 扶墙脚部 student checkpoint。评估不会上传 W&B。

## Checkpoint 下载与环境

先在评估 Python 环境中运行 `python -m wandb login`，或使用已有的 `WANDB_API_KEY`。

也支持 `https://wandb.ai/<entity>/<project>/runs/<run-id>` 和 `run:<entity>/<project>/<run-id>`。链接的查询参数会被忽略。默认优先下载 `checkpoint_final.pt`（也识别 `.ckpt` / `.pth`），否则选择编号最大的 checkpoint；多个同优先级文件时需用 `--wandb-file` 指定 run Files 中的完整相对路径，例如 `--wandb-file checkpoint_2000.pt`。这里只读取训练脚本通过 `run.save` 上传的 Files。

下载缓存在 `.cache/wandb-checkpoints/`，以 API 地址、run、文件名和远端摘要区分，远端文件变化后重新下载；可用 `--checkpoint-cache` 修改目录。再次运行仍需联网读取文件元信息。W&B API 地址沿用 SDK 配置（默认 `https://api.wandb.ai`）；私有部署使用 `WANDB_BASE_URL` 和 `run:` 格式。

本地 `--checkpoint /path/to/checkpoint_final.pt` 继续可用。不传参数仍选择本地 pipeline。`PYTHON=/path/to/python` 可指定安装了 Isaac Lab / PyTorch / wandb 的 Python，`ISAACLAB_PATH` 指向 IsaacLab 仓库。仍需准备 `artifacts/foot_reach/workspace.npz`、`manifest.json` 及 checkpoint 配置引用的数据和机器人资产；下载权重不会下载这些资源。

## 目标、指标与输出

JSON 格式：`[[0.1, 0.15, -0.5], [0.2, 0.15, -0.4]]`。这里只是格式示例，不保证这些点可达。目标为左脚脚底中心相对**实时 root 完整坐标系**的位置；X 前、Y 左、Z 上，单位米。因此机器人 root 移动时，其对应世界目标也会移动。程序额外记录支撑脚、手、root 的世界位置漂移。

默认从 `workspace.npz` 的完整可行点集中选择目标：从初始脚位置出发，每次选距离上一个目标最近的未选点，连续执行 20 个不同点，排除初始点本身。不再随机采样，也不限于 validation 点；`--seed` 只控制仿真随机性。候选不足时明确报错，不重复凑数。这是局部最近邻路径，不保证覆盖整个可行域或全局最短路径。`--targets` 指定的目标仍严格按文件顺序执行。整个序列只在开始时重置一次并稳定 `--settle 1` 秒，随后连续直接切换脚 EE target；每个点只执行一次，留出 `--reach 2` 秒到达时间，继续保持目标 `--hold 1` 秒。没有插值轨迹；仅统计最后 hold 窗口内每帧的三维位置误差，`mean_error_m` 是该窗口的平均欧氏距离（越小越好），不是阈值命中率。推理使用确定性 student action 和 checkpoint 观测归一化参数。

成功/失败仅表示是否稳定，与 reaching 误差大小无关。每个目标的 reach、hold（首个目标还包括初始 settle）未触发失稳才算成功。失稳时立即停止整个序列，不重置后继续。成功率/失败率以已尝试目标为分母，剩余目标单独记录为 `unattempted_targets`，不计作成功或失败；`sequence_completed` 表示是否稳定完成全部目标。失败 trial 保留在分母中；不足完整 hold 的误差只作为部分窗口记录，不参与每个点的完整窗口均值。

输出目录默认 `artifacts/foot_reach_eval/<时间戳>/`；可用 `--output` 指定新目录：

- `config.json` / `targets.json`：checkpoint、种子、时间与坐标约定、目标。
- `samples.csv`：逐控制步的目标/实际坐标、欧氏距离误差、各支撑点漂移与失败原因。
- `metrics.csv`：每个 trial 的指标表，可直接用表格软件打开。
- `summary.json`：每个点每次重复的平均误差、RMSE、P95、最大误差、三轴 MAE，以及整体稳定成功率/失稳率。单位米，乘 1000 为 mm。
- `evaluation.mp4`：10 FPS 的真实 Isaac 画面；红球为指令目标，绿球为实际脚底中心，并显示误差。视频按仿真时间播放。

稳定性每个控制步检查，满足任一条件立即停止序列并将当前目标标记失败：

- root 相对序列开始时 reset 的世界位置漂移 > 0.20 m；
- root 坐标系下归一化重力的 Z 分量 > -0.7，约对应相对直立倾斜超过 45.6°；
- 右支撑脚 body 相对序列开始时 reset 的世界位置漂移 > 0.12 m；
- 左扶墙手 body 相对序列开始时 reset 的世界位置漂移 > 0.15 m；
- 左脚位置或 root state 出现 NaN / Inf。

这些是姿态和漂移代理指标，没有直接检查接触力、手是否仍贴墙或支撑脚是否承重。稳定成功并不表示已经准确到达目标；精度需单独看误差报告。网页仍采用限速平滑移动，失稳自动暂停。

## VS Code 远程网页

运行 `--web` 后，等待终端出现 `WEB_READY`。在 VS Code 的 **Ports / 端口** 面板转发远程 `8765` 端口，然后在本地浏览器打开 `http://localhost:8765`（若 VS Code 分配了其他本地端口，则使用该端口）。默认仅监听服务器 `127.0.0.1`。

页面旁边提供 X/Y/Z ± 方向按钮、暂停/继续、重置。每次点击 1 cm，指令以最多 0.1 m/s 平滑移动；红球显示正在执行的当前目标。控制方向按机器人坐标系，不按摄像机屏幕坐标。失稳自动暂停，点击重置恢复。目标仅限制在已采样点的包围盒内，盒内不保证所有点都可达。

HTTP 线程只接收指令/传输图像，Isaac 和推理始终在主线程运行。画面采用 JPEG 轮询，无浏览器插件需求。界面力求实时，实际帧率受服务器渲染速度影响。`Ctrl+C` 停止；`--web-seconds 30` 可运行有限时长用于测试。网页模式不产生固定目标 trial 的精度报告，显示的是实时误差。

### 失稳后继续运行

加 `--continue-on-instability` 可关闭失稳触发的提前停止，仍记录失败原因，并连续跑完所有目标，不重置。完整 hold 窗口的误差仍会记录，包括失稳期间的数据；查看精度时需结合 `failed` / `success` 判断。`all_targets_evaluated` 表示所有窗口均已执行，`sequence_completed` 仍要求全程稳定。此选项用于固定点评估，网页失稳暂停逻辑不变。
