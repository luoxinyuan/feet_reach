# 扶墙脚部固定点评估 / 网页遥操作

在项目根目录运行。默认使用 `outputs/wall-foot-reach/latest_pipeline.txt` 指向的最终 finetune checkpoint；可用 `--checkpoint /absolute/path/checkpoint_final.pt` 指定。只接受包含观测归一化参数的 student checkpoint。评估不会修改训练文件或上传 WandB。

```bash
# 默认 7 个固定目标 × 3 次重复，并录像
bash scripts/eval_wall_foot_reach.sh --video

# 自定义目标（米），可从上次输出的 targets.json 修改
bash scripts/eval_wall_foot_reach.sh --targets /path/targets.json --repeats 5 --video

# 网页遥操作，可同时录像
bash scripts/eval_wall_foot_reach.sh --web --video --port 8765
```

JSON 格式：`[[0.1, 0.15, -0.5], [0.2, 0.15, -0.4]]`。这里只是格式示例，不保证这些点可达。目标为左脚脚底中心相对**实时 root 完整坐标系**的位置；X 前、Y 左、Z 上，单位米。因此机器人 root 移动时，其对应世界目标也会移动。程序额外记录支撑脚、手、root 的世界位置漂移。

默认固定目标为原始姿态 + 验证集端点的确定性最远点采样，避免只测训练目标。每个 trial 从原始姿态重置，默认稳定 1 秒、平滑过渡 2 秒、保持 3 秒。仅统计保持阶段的位置误差。推理使用确定性 student action，仅输入归一化 policy 观测，没有 teacher joint labels 或 privileged observations。

输出目录默认 `artifacts/foot_reach_eval/<时间戳>/`；可用 `--output` 指定新目录：

- `config.json` / `targets.json`：checkpoint、种子、时间与坐标约定、目标。
- `samples.csv`：逐控制步的目标/实际坐标、欧氏距离误差、各支撑点漂移与失败原因。
- `metrics.csv`：每个 trial 的指标表，可直接用表格软件打开。
- `summary.json`：每个点每次重复的平均误差、RMSE、P95、最大误差、三轴 MAE、阈值内比例，以及整体成功率/失败率。单位米，乘 1000 为 mm。
- `evaluation.mp4`：10 FPS 的真实 Isaac 画面；红球为指令目标，绿球为实际脚底中心，并显示误差。视频按仿真时间播放。

默认成功定义：完整完成保持阶段，且 >=95% 的保持帧误差 <=3 cm（`--threshold 0.03`）。失稳立即结束 trial，不做自动重置掩盖失败；部分保持阶段的误差仍保留，但明确标记失败。失稳阈值：root 世界漂移 >20 cm、root 重力 Z 分量 >-0.7、支撑脚漂移 >12 cm、扶墙手漂移 >15 cm 或非有限状态。这些为评估阈值，不是物理接触力成功判据；低误差也不能单独证明有足够墙面支撑力。

## VS Code 远程网页

运行 `--web` 后，等待终端出现 `WEB_READY`。在 VS Code 的 **Ports / 端口** 面板转发远程 `8765` 端口，然后在本地浏览器打开 `http://localhost:8765`（若 VS Code 分配了其他本地端口，则使用该端口）。默认仅监听服务器 `127.0.0.1`。

页面旁边提供 X/Y/Z ± 方向按钮、暂停/继续、重置。每次点击 1 cm，指令以最多 0.1 m/s 平滑移动；红球显示正在执行的当前目标。控制方向按机器人坐标系，不按摄像机屏幕坐标。失稳自动暂停，点击重置恢复。目标仅限制在已采样点的包围盒内，盒内不保证所有点都可达。

HTTP 线程只接收指令/传输图像，Isaac 和推理始终在主线程运行。画面采用 JPEG 轮询，无浏览器插件需求。界面力求实时，实际帧率受服务器渲染速度影响。`Ctrl+C` 停止；`--web-seconds 30` 可运行有限时长用于测试。网页模式不产生固定目标 trial 的精度报告，显示的是实时误差。
