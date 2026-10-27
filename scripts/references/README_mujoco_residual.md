# SONIC / CEER residual 与 Jacobian policy 的 MuJoCo 交互仿真

运行时只有已训练 student 和 frozen low-level，不读取 AMASS、teacher、motion reference 或下一帧目标。从站立开始，通过鼠标给上半身施力。

## 安装与启动

使用 Python 3.10/3.11，安装现有独立 demo 的依赖：

```bash
python -m pip install -r deploy/mujoco_whole_body/requirements.txt
python scripts/mujoco_whole_body_web.py --policy sonic --port 8776
# 另一终端：
python scripts/mujoco_whole_body_web.py --policy ceer --port 8777
# Jacobian compliance policy：
python scripts/mujoco_whole_body_web.py --policy jacobian --port 8778
```

打开 http://localhost:8776 或 http://localhost:8777。VS Code Remote SSH 用户转发相应端口，然后在自己电脑的浏览器打开。默认只绑定 localhost。

SONIC 首次启动自动从 NVIDIA 官方 Hugging Face 仓库下载两个 ONNX，合计约 88 MB。版本固定为 `bfd764a`，下载完成后校验 SHA-256，再原子写入 `.cache/mujoco-sonic/`。以后启动无需下载。CEER 及两版 student 权重已随仓库提供。无需 CUDA、Isaac Lab、AMASS、训练 checkpoint、SONIC 源码或 Hugging Face 登录。

网页左上角显示当前策略；默认暂停于站立状态。点击开始，用鼠标左键拖拽 torso、肩、肘、手腕或手部；力上限 30 N，有外力箭头。松手释放，空格暂停，R 复位，F 切换相机跟随。浏览器断开自动暂停。

原版直接控制 student 保持为默认 `--policy direct`。桌面 GUI 也支持同样的策略选择：

```bash
python scripts/mujoco_whole_body_student.py --policy sonic
python scripts/mujoco_whole_body_student.py --policy ceer
```

无图形环境可运行数值检查：

```bash
python scripts/mujoco_whole_body_student.py --policy sonic --headless --seconds 20
python scripts/mujoco_whole_body_student.py --policy ceer --headless --seconds 12 --test-force
```

第二条在 5–7 秒给左手施加 15 N 世界 +X 拉力。可用 `--report /path/result.json` 保存结果。

## Jacobian policy

`--policy jacobian` loads the Jacobian compliance checkpoint exported to
`deploy/mujoco_whole_body/jacobian/`. It uses the checkpoint's 381-dimensional
residual actor, frozen observation normalization, and the exact frozen CEER
low-level controller referenced by that run. Dragging applies the same point forces
and force observation used by the other interactive policies. The checkpoint hash
and source run are recorded in the bundle metadata. Export from a training
environment using the Jacobian checkpoint recorded in that bundle's metadata.

## 控制器接口

Student 以 50 Hz 运行，输入 381 维：角速度、重力方向、5 帧关节角、3 帧 residual 输出、9 个施力点的实际本体系力、关节速度，以及自己生成的关节参考 / 跟踪误差 / 参考更新速度。归一化使用各自训练完成后的固定统计。

Student 的原始 29 维输出用于 `q_ref += 0.02 * 2 * tanh(action)`，遵守训练时的软关节限位及 0.6 rad 跟踪偏差约束。零 residual 严格保持参考，reset 从实际站姿关节角初始化；网络本身仍可能在零外力时输出非零 residual。

- SONIC：官方 encoder + decoder ONNX；10 帧自生成关节参考（速度输入为零）、当前朝向下的直立参考，以及原版 10 帧历史。保留 SONIC 电机增益和 action scaling。
- CEER：原通用 finetune 的 actor / 状态 estimator 冻结，joint predictor 不加载；5 帧接口直接填入自身 q_ref，并转回原 AMASS 关节顺序。双手接口由同一 q_ref 做 USD 几何 FK；root 接口只用实际高度、速度、当前朝向。保留 CEER action scaling、平滑、延迟及 boot protection。力条件通道固定为 5，实际鼠标力上限仍为 30 N。
- Jacobian：使用 Jacobian checkpoint 导出的 381 维 student、对应 VecNorm，以及该 checkpoint 引用的冻结 CEER low-level；参考积分、CEER joint order、FK 接口、力通道和 actuator 参数均从该 checkpoint 配置导出。

两版物理步长 5 ms、控制周期 20 ms；默认站姿、高度、PD、armature、扭矩限制、关节限位均从训练资产导出。CEER 禁止自身碰撞，SONIC 保留模型自身碰撞。接触几何来自随附 MuJoCo G1 XML，接触求解器与训练的 Isaac 不同，不保证轨迹完全一致。

## 已验证结果

- 导出的 residual actor 对原生训练 actor：最大绝对误差约 2.4e-6。
- portable 对 native frozen low-level：SONIC < 7e-7，CEER 为 0；参考积分误差 < 5e-7；reset 清除历史和 residual 状态。
- CEER 手部 FK 与训练 USD FK 对齐；具体误差见 CEER metadata。
- 两版分别无力运行 20 秒、左手 15 N 拉拽 12 秒，状态有限，root 高度未触发跌倒检测（z < 0.4 m）。
- Chromium 实际测试通过：选点、拖拽、30 N 限幅、松手归零、暂停 / 复位、缩放后选点、连接互斥和断线暂停。测试服务器实时倍率约 1.00。

**当前策略不保证无力静止或松手即停。** 本次 MuJoCo 无力 20 秒，SONIC 最终平移约 0.08 m，CEER 约 1.62 m。此处没有增加额外站立控制器或掩盖漂移的复位逻辑。短时测试不代表任意拖拽都稳定，跌倒可按 R 复位。

发布的机器可读验证记录位于 `deploy/mujoco_whole_body/validation_residual.json`。归一化和模型哈希见各 bundle 的 `SHA256SUMS.json`；SONIC 官方模型哈希及下载地址见其 `metadata.json`。

## 分段施力版（segmented，18 个候选点）

新增 `--policy segmented`，对应已完成训练的分段 Jacobian CEER residual
checkpoint `full_20261024_122951 / 00lci6cu / checkpoint_final.pt`，不是正在训练的 AMP 版。
权重随仓库发布，无需 Isaac、AMASS、施力缓存或训练 checkpoint。

```bash
pip install -r deploy/mujoco_whole_body/requirements.txt
# 本机桌面窗口
python scripts/mujoco_whole_body_student.py --policy segmented
# 或浏览器交互；远程 VS Code 转发 8779 端口
python scripts/mujoco_whole_body_web.py --policy segmented --port 8779
```

浏览器打开 `http://127.0.0.1:8779`。从站立暂停状态开始，空格开始/暂停，
鼠标左键拖动上半身，R 复位。鼠标选中最近的候选施力点，箭头显示实际外力，
单点上限 30 N。支持胸、背、躯干两侧、左右上臂/前臂、肩肘腕手和一个头部中心点。
点击头部几何会映射到头部中心；不把额外力矩直接输入 student。

该版输入 408 维，其中 18×3=54 维为各点实际施力的机器人本体系表示，
其余仍为自身状态、历史输出和自己生成的参考。没有下一帧目标或 motion reference。
相同刚体上的多个点保留独立观测通道；仿真合并它们的力和关于质心的力矩。
冻结 CEER 接收 residual 更新的关节参考，所有网络在 CPU 推理。

验证：导出 actor 与训练模型最大误差 2.39e-6，冻结 CEER 导出误差为 0；
18 点逐点输入、位置选择、合力/力矩、广义力以及复位检查通过。
MuJoCo 无外力运行 20 秒未跌倒，最终水平位移约 5.3 cm；
5–7 秒给左手施加 15 N 的测试在约 7.17 秒跌倒。
因此可用于交互对比，但此版不保证任意拖拽稳定，也不保证松手保持。
验证记录见 `deploy/mujoco_whole_body/validation_segmented.json`。

### Multi10 单手多方向力序列

新增独立导出包 `deploy/mujoco_whole_body/20261026-segmented-multi10/`，来自
multi10 训练 run `4vqahv90` 的最终 checkpoint。它和较早的 `segmented`
版本并存，不覆盖旧 policy。桌面和浏览器交互 demo 均可选择：

```bash
python scripts/mujoco_whole_body_student.py --policy segmented-multi10
python scripts/mujoco_whole_body_web.py --policy segmented-multi10 --port 8781
```

复现单机器人正面施力演示视频：

```bash
python scripts/mujoco_whole_body_force_demo.py \
  --policy segmented-multi10 \
  --output artifacts/20261026-mujoco-multi10-force-demo/rollout.mp4
```

视频固定 reference 姿势，逐段对左手施加 5/10 N 左右向、8 N 上下向和 12 N 前向外力；
黄色箭头从左手施力点出发。MuJoCo 使用导出的 multi10 student 与冻结 CEER，
不加载 Isaac、训练数据或 force cache。MuJoCo 与 Isaac 的动力学不同，力序列演示
用于直观检查，不代表 Isaac 评测指标的复现。

### 静止轨迹增强版

最近一次评测的 static-hold augmented checkpoint 已单独打包为
`deploy/mujoco_whole_body/segmented-static-hold/`，保留原 `segmented/` 版本。
桌面运行：

```bash
python scripts/mujoco_whole_body_student.py --policy segmented-static-hold
```

浏览器运行：

```bash
python scripts/mujoco_whole_body_web.py --policy segmented-static-hold --port 8780
```

对应 checkpoint 为 `static_hold_aug_20261025_165158 / 3dlvh2d9`。该 policy
在单手施力后被动手臂抬高超过 10 cm 的比例由 13.7% 降至 4.3%；左右侧表现不完全一致，
详细评测见 `artifacts/ceer_segmented_singlehand_stop_eval_20261025/README.md`。
