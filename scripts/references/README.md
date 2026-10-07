# G1 按人类照片适配的侧身扶墙抬膝 reference

从本项目 `active_adaptation/assets/G1/g1_29dof_rev_1_0_flat.usd` 直接读取
29 个关节、限位、惯量和外观网格，无需启动 Isaac Sim，也不使用替代 URDF。

## 复现

已在项目独立虚拟环境 `.venv-reference` 中安装依赖，不影响训练环境。
从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 .venv-reference/bin/python scripts/references/generate_wall_support.py
```

重新创建环境时，使用 Python 3.13（本次验证版本）：

```bash
python3.13 -m venv .venv-reference
.venv-reference/bin/pip install -r scripts/references/requirements-lock.txt
```

锁定 Crocoddyl 3.1.0 / Pinocchio 3.8.0 及对应 Boost、URDFDOM、TinyXML
二进制依赖；不要单独升级其中一个包。

## 姿态和方法

- 右脚支撑，脚底目标 `(0.00, -0.10, 0.00)` m。
- 左膝向前抬，脚底目标 `(0.17, 0.07, 0.22)` m，即扶墙手同侧的腿。
- 左手接触 `(0.028, 0.25, 1.24)` m，手指朝上，肘部弯曲，右手自然下垂。
- 墙在身体左侧，内侧面 `y=0.25` m，墙高 1.5 m。
  对应场景为 `cfg/objects/wall_support_reference.yaml`。
- 世界坐标：X 向前、Y 向左、Z 向上，距离为米，关节角度为弧度。

参考了以下实拍照片，人工提取“侧身靠墙、躯干直立、支撑腿较直、膝盖向前抬、
扶墙肘自然弯曲”的姿态特征，再根据 G1 的肢体比例和关节限位优化。
不是从单张图片精确恢复人体三维关节角，也不声称逐点复现照片。

- 整体姿态：[LiveUp — standing hip march](https://www.liveup.org.au/resources/strength-exercise/hip-exercises-for-older-people)
- 扶墙手臂：[单手扶墙单脚站立实拍](https://www.hiza-seitai-mizuharu.jp/blog.kataashidachi)

本地参考图在 `artifacts/wall_support/human_reference/`，来源仍归原网站。

Crocoddyl `SolverDDP` 优化一个配置空间步长，terminal cost 使用
`ResidualModelFramePlacement`、`ResidualModelCoMPosition`、关节限位 barrier
和姿态正则化。该步长是关节配置增量，不是电机控制输入；生成的是单个静态
reference，不是从站立进入此姿态的运动轨迹。手部允许在墙面平面内转动，
同时约束掌面法向、接触位置，并对手腕设置中立姿态偏好与 ±0.55 rad barrier。
另加躯干/骨盆直立、腰部中立和支撑腿接近伸直的偏好。输出再次验证三个手腕角度
均小于 32°；当前屈伸约 23°，其余两个轴约 1.5°。

之后通过 SciPy SLSQP 求右脚 6D 接触力矩和左手 3D 接触力，约束包括
浮动基静力平衡、单边接触、摩擦锥、脚底压力中心和 USD 电机力矩限位。
再把求得的电机力矩送入 Crocoddyl `DifferentialActionModelContactFwdDynamics`，
检查零速度时的加速度接近零。依赖摩擦系数假设：地面 0.7、墙面 0.6。

使用原始网格顶点验证脚底离地、地面和左侧墙不穿透（数值容差 0.1 mm）；
预览网格仅为显示简化。
未进行全身自碰撞检测或 Isaac Lab 策略跟踪测试。USD 的三个 5 g mimic marker
未显式提供惯量，按原点处点质量处理。完整诊断写入 JSON。

接口参考：[Crocoddyl frame-placement residual](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/classcrocoddyl_1_1ResidualModelFramePlacementTpl.html)。

## 输出

文件保存在 `artifacts/wall_support/`（该目录沿用项目的 gitignore 规则）：

| 文件 | 内容 |
| --- | --- |
| `reference.png` | 真实 USD 网格双视角图片 |
| `reference.html` | 可旋转、缩放的独立离线三维预览 |
| `joint_angles.csv` | 每个关节名称、弧度、角度、静态力矩 |
| `reference.json` | 完整 reference、基座位姿、接触力、验证结果 |
| `reference.npz` | NumPy 可读取的单帧 reference |

NPZ 字段：`joint_names` (29,), `joint_pos` (1,29), `joint_vel` (1,29),
`root_pos` (1,3), `root_quat_xyzw` / `root_quat_wxyz` (1,4),
`q_pinocchio` (36,), `torque` (29,)。根位姿不能省略，否则不会得到同一姿态。
这是独立静态 reference 格式，未伪装为项目运动数据集格式。

应用于 Isaac Lab 时按名称重排，不要假定关节数组顺序相同：

```python
import numpy as np
ref = np.load("artifacts/wall_support/reference.npz", allow_pickle=False)
lookup = {name: i for i, name in enumerate(ref["joint_names"].tolist())}
# robot.joint_names 为接收端实际关节顺序。
q_reference = ref["joint_pos"][0, [lookup[name] for name in robot.joint_names]]
base_position = ref["root_pos"][0]
base_quaternion_wxyz = ref["root_quat_wxyz"][0]
```

连续脚部 reach 数据增强与 low-level 训练见 [FOOT_REACH.md](FOOT_REACH.md)。
