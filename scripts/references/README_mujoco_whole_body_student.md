# Whole-body student：本地 MuJoCo / 浏览器交互

这是一个独立 demo：**只运行训练好的 student，从默认站姿开始，用鼠标给上半身施力。**
仓库已经包含导出的 student、观测归一化参数、G1 XML 和全部所需 mesh。
无需安装 Isaac Lab、下载 AMASS、重新训练，或克隆其他仓库。模型与权重约 25 MB，直接通过 Git 下载，不需要 Git LFS。

## 在自己的电脑安装

建议 Python **3.10 或 3.11**，使用独立虚拟环境。不需要对整个训练项目执行 `pip install -e .`。

```bash
git clone https://github.com/luoxinyuan/feet_reach.git
cd feet_reach
python -m venv .venv
```

激活环境：

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
# 如果 PowerShell 不允许激活，可直接用 .venv\Scripts\python.exe 执行下文命令。
```

然后安装 demo 依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r deploy/mujoco_whole_body/requirements.txt
```

## 浏览器版（推荐先用这个）

```bash
python scripts/mujoco_whole_body_web.py --port 8765
```

打开 **http://localhost:8765**，点击“开始 / 继续”。初始画面暂停在站姿。

在远程服务器运行时，使用 VS Code **Ports / 端口 → 转发 8765 → 在浏览器中打开**。服务默认仅监听 `127.0.0.1`。
Linux/macOS 也可以使用 `bash scripts/run_mujoco_whole_body_student.sh --web --port 8765`。
Shell 入口使用当前环境的 `python`，也可用 `COMPLIANCE_PYTHON` 指定解释器。

- 左键拖拽胸部、手臂或手：施加引导力；松手后卸力。
- 橙色箭头显示实际外力，青色点标记所选施力点；页面显示牛顿数。
- 右键拖拽：转动视角；Shift + 右键平移；滚轮缩放。
- **R** 恢复站姿；**空格**暂停/继续；**F** 开关相机跟随。
- 浏览器切后台会取消拖拽并暂停；失去心跳 3 秒后也会暂停。重新连接后需主动继续。
- 每个实例只允许一个标签页控制，其他标签页会提示等待控制权。
- 终端 **Ctrl+C** 停止服务。

**渲染后端：**Windows 使用 GLFW，macOS 使用 CGL；Linux 桌面使用 GLFW，无桌面服务器使用 EGL。student 始终在 CPU 上推理，OpenGL 使用本机显卡，不要求 NVIDIA CUDA。Linux 服务器的 EGL 实测占用约 150 MiB 显存。

如果 Linux 无桌面环境无法初始化 EGL，可以安装系统 OSMesa 库后使用软件渲染：

```bash
MUJOCO_GL=osmesa python scripts/mujoco_whole_body_web.py --width 640 --height 400 --fps 10
```

默认画面为 960×600 / 20 fps；仿真为 500 Hz，策略为 50 Hz。低性能电脑可以降低分辨率和帧率，页面会显示实际仿真速度倍率。
**目前已在 Linux 验证，Windows/macOS 做了后端与路径适配，但尚未在这两种系统实机验证。**

## 原生窗口与无窗口检查

桌面窗口：

```bash
python scripts/mujoco_whole_body_student.py
```

数值检查，不需要窗口或 GPU：

```bash
python scripts/mujoco_whole_body_student.py --headless --seconds 30
python scripts/mujoco_whole_body_student.py --headless --seconds 30 --test-force
```

第二条命令在第 5–7 秒给左手施加 15 N 世界坐标 +X 拉力。可加 `--report result.json` 保存结果。

## 施力与 policy 接口

鼠标点击上半身后，会选择该侧最近的训练施力点（torso、shoulder_roll、elbow、wrist_roll、hand_mimic），并在点击时的屏幕平面中移动弹簧锚点。一次拖拽一个点，腿、脚和 pelvis 不接受施力拖拽。

`F_des = 80 * (mouse_anchor - point_position) - 4 * point_velocity`。
每点上限 **30 N**，变化率最多 **150 N/s**，松手后最多约 0.2 秒衰减到零。暂停时立即清除残留外力。力在施力点产生自然力矩，不使用旋转目标控制。

- 网络为 `adapt_module + adapt_joint_module + actor_student` 的确定性输出。内部 joint predictor 属于 student，不读取外部 target。
- 265 维观测：本体角速度 3、重力方向 3、绝对关节角历史 145、上三次动作 87、9 个上半身点的本体系实际外力 27。
- 使用训练 checkpoint 的冻结归一化；物理仿真与外力观测使用同一个限幅后的力。
- 关节动作是默认关节角 + `0.5 * clip(action, -10, 10)`，关节顺序、PD、armature、力矩上限见随附 metadata。
- 初始 root 高度 0.74 m、默认站姿、零速度；没有额外的站立或停止控制器。
- 模型和策略默认从 `deploy/mujoco_whole_body/` 加载。`--model` / `--bundle` 可覆盖默认路径。
- 手部力点使用训练 USD 的 FixedJoint 偏移（wrist_yaw 局部 x=0.116 m），运行时添加到模型。

## 验证与局限

- 导出的 TorchScript 与原生 PPO student：20 组随机观测输出最大误差 **0**，输入只有 `policy`。
- Linux MuJoCo 无外力 30 秒没有检测到跌倒（检测条件 root z < 0.4 m），但位置漂移约 **1.55 m**。
- 第 5–7 秒施加 15 N 拉力、总计 30 秒：没有检测到跌倒，但结束时离起点约 **4.90 m**。实际外力的坐标转换误差 < 5e-7 N。
- 原生 GUI 和 Chromium 中验证了选点、拖拽、松手归零、复位、画面缩放后选点，以及浏览器断线暂停；浏览器仿真实时倍率约 1.00。

**当前 student 不保证原地静止，也不保证松手即停。** 这些是短测，不能代表任意拖拽都稳定；MuJoCo 与训练所用 Isaac 的接触和积分也有差异。跌倒后会提示，按 R 复位。

## 导出与来源

权重、机器人资产来源和 SHA-256 校验信息见 [部署包说明](../../deploy/mujoco_whole_body/README.md)。
完整训练 checkpoint 没有上传；运行 demo 不需要它。

如需从已有训练 checkpoint 重新导出：

```bash
python scripts/export_whole_body_student.py --checkpoint /path/to/checkpoint_final.pt --output /path/to/export
python scripts/mujoco_whole_body_web.py --bundle /path/to/export
```

导出需要训练 checkpoint 使用的 Python 序列化依赖。原生 PPO 对照检查还需要完整训练环境；这些依赖不属于 demo 安装要求。
