# Passive whole-body locomotion：AMASS 离线最小验证

本阶段完成 **AMASS transition 采样、双手外力生成、接触逆动力学与逐帧检查**。未启动 locomotion 训练，也未声称已经在仿真中学会行走。沿用上一实验的假设：外力只作用于双手 EE，不直接注入外力矩。原 passive compliance 训练不受影响。

## 1. 数据和 transition

本地 `/home/xl521/hiearchical_compliance/dataset/amass` 是已重定向到 G1 的数据：8,703 个 clip、4,916,587 帧、29 关节。不是直接将 SMPL 人体惯性用于机器人。质量、惯性、关节顺序及手脚几何由当前 G1 USD 构造 Pinocchio 模型，机器人质量约 33.36 kg。

数据没有保存 fps 和原始文件名。按仓库处理代码假定 50 Hz，并以 root 位移差分/已存速度交叉验证（中位比值约 1.0006）。保存原 clip ID 和起止帧；clip 级 train/val 划分仅用于初步实验，无法保证来自同一原始 AMASS motion 的不同 clip 不跨集合。

随机采样 0.4/0.8/1.2 s 的同 clip 正向时间片段，平衡直行、左转、右转、hold；过滤明显倾斜、过快运动、关节超限及没有合理支撑脚的片段。float16 姿态先转 float64，经 11 帧三阶 Savitzky–Golay 平滑，再通过 `pin.difference` 求速度和加速度。仅做一次全片段地面高度平移，不逐帧钉住脚底高度。

**保留完整 q/v/a 与步态相位**，不能把后一个 mid-swing pose 作为静态目标。难例测试 1/1.5/2/3 倍时长：v 缩放为 v/s，a 为 a/s²。减速不保证改善可行性，因为原动作可能依赖动态平衡。

## 2. 为什么两个 target 不能唯一确定外力

需要指定转移时长、速度、接触时序和动作路径。质心动量方程为：

```
dot(p) = m*g + F_ground + F_hands
dot(L_COM) = M_ground_COM + M_hands_COM
```

地面承担支撑和部分推进/转向，不能令手部力直接等于 m*a，或把 root yaw 的惯量乘角加速度直接当成全部外力矩。这里用 Pinocchio 全身 centroidal momentum / RNEA 计算真实动力学项，虚拟惯量只定义引导信号。

对两手施加纯力，所有量在世界坐标表达，力矩统一关于实时 COM：

```
w_hands = A*f
A = [ I                 I              ]
    [ skew(p_left-c)    skew(p_right-c) ]
f = [f_left; f_right]
```

双手不重合时 A 通常秩为 **5**。令 d=p_left-p_right，必须满足：

```
d · (M_COM - (p_right-c) × F) = 0
```

因此不能任意指定六维 wrench。双手反向力可以形成 yaw 力偶，但独立力偶不能沿双手连线。严格任意 SE(3) wrench 需要增加不共线施力点，或允许直接 EE torque。

## 3. 推荐的引导规则：先做 SE(2) 行走与转向

保留水平位移和 yaw 为主要引导量，竖直、roll/pitch 由站立平衡处理。参考私有，student 不读它：

```
F_xy* = Mv*a_ref_xy + Bdrive*v_ref_xy
        + Kp*(c_ref_xy-c_xy) + Dv*(v_ref_xy-v_xy)
Mz*   = Jv*alpha_ref + Byaw*omega_ref
        + Kpsi*wrap(psi_ref-psi) + Dpsi*(omega_ref-omega)
```

起始参数：Mv=4 kg，Bdrive=35 Ns/m，Jv=0.5 kg·m²，Byaw=6 Nms/rad。离线计算在理想 reference 状态上，反馈误差为零。Kp=60 N/m、Dv=12 Ns/m、Kpsi=8 Nm/rad、Dpsi=2 Nms/rad 可作为后续仿真起点，**尚未调参验证**。

`Bdrive*v_ref` 和 `Byaw*omega_ref` 使匀速行走/匀速转向仍有可辨识的外力信号。只用 m*a 会在匀速时产生零引导力，使 student 难以区分“继续走”和“停止”。这是一种人为指定的交互协议，不是声称 AMASS 中存在唯一真实拉力。

先构造保持 F 和 Mz、兼容双手几何的 nominal 六维 wrench：在双手中点施加合力，再用与双手连线正交的最小范数力偶补足所需 yaw。这样保留不可避免的 roll/pitch 分量，不将它们错误设为零。双手太近或连线接近竖直时拒绝。

随后联合优化手部力 f、足底接触 wrench λ，隐式求关节力矩 τ：

```
min   ||W*(A*f-w_nominal)||² + εf*||f||²
      + εlambda*||lambda||² + εsmooth*||f-f_prev||²
s.t.  M(q)*a+h(q,v) = Sᵀ*tau + J_handᵀ*f + J_footᵀ*lambda
      (A*f)[Fx,Fy,Mz] = [Fx*,Fy*,Mz*]      # 推荐模式，硬约束
      接触法向力非负、摩擦锥、CoP/扭转摩擦边界
      摆动脚 lambda=0，|tau| <= 0.85*effort_limit
      ||f_left/right|| <= 60 N
      ||f_t-f_prev||/dt <= 500 N/s         # 逐帧验证时启用
```

摩擦系数 0.7，足底 CoP 半宽/半长 0.025/0.045 m，扭转比 0.015 m；摩擦/力范数/变化率使用保守多面体近似，CoP 轴随脚部 yaw 旋转。输出完整六维 `A*f`，连同地面 wrench 验证全部六维动量平衡，而不只是比较水平力大小。允许竖直力及 roll/pitch 补偿，但与 nominal 的合力偏差 >20 N 或合力矩偏差 >5 Nm 则拒绝标签。hold 强制两手 f=0。

同时提供两种诊断模式：严格六维 wrench 硬匹配，以及所有引导量均软约束的 relaxed QP。后者可能主要产生平衡补偿，**不直接作为推荐训练标签**。

## 4. 连续训练如何接入

- Teacher 输入 reference 和本体观测，用 dense whole-body motion tracking；student 输入本体历史、历史 action、实际 EE 力及其历史，不输入 joint/EE/root target、yaw target、anchor 或 transition phase。
- 隐藏 reference 在片段内按时间推进；片段末按接触状态、速度连续性选下一片段，不强迫每个目标停下。hold 单独采样可零外力站稳的片段。
- 每个物理步由当前 COM/双手位置重建 A、计算反馈引导 wrench、联合求解并施力。力向量转为实际 root/body 坐标供 student 使用；不能按目标 heading 旋转观测。
- 接触切换须通过实际接触传感器确认；参考 contact 只供 teacher/外力生成器使用。QP 失败应停止该参考/重采样，而不是伪造一个满足了目标的截断力。
- Student reward 可沿用 teacher 的同一 dense reference；若要求模仿 teacher 实际执行轨迹，则先采集 teacher rollout。两者需在实验中明确区分。

## 5. 实现与复现

```bash
.venv-reference/bin/python scripts/references/sample_passive_locomotion.py
.venv-reference/bin/python scripts/references/refine_passive_locomotion.py --mode planar
.venv-reference/bin/python scripts/references/refine_passive_locomotion.py --mode relaxed
.venv-reference/bin/python scripts/references/validate_passive_locomotion.py
.venv-reference/bin/python -m unittest discover -s tests -p test_locomotion_wrench.py
```

输出在 `artifacts/passive_locomotion/`：原始采样与严格匹配 `report.json`、推荐规则 `planar_report.json`、软匹配诊断 `relaxed_report.json`、逐帧复查 `validation.json`、可查看的外力实例 `force_examples.csv`、通过检查的 `validated/*.npz`。初轮 96 样本留在 `artifacts/passive_locomotion_pilot96/`。

逐帧检查去除片段两端各两帧，避免差分边界加速度；变化率约束从该验证片段的第二帧开始，不代表两个不同片段的边界也已平滑。

**限制：**当前是接触逆动力学的离线可解性检查。支撑由足底高度 <3.5 cm、速度 <0.35 m/s 估计，平足接触及保守 CoP 为近似；未约束严格无滑动接触加速度、未做全身碰撞检查、未进行闭环仿真。通过不代表真实接触轨迹可执行，更不代表无源性证明。训练前应进行接触约束下的轨迹修正及 Isaac rollout 筛选。

参考：[Pinocchio 官方动力学教程](https://gepettoweb.laas.fr/talks/jnrh2023/presentation%20-%20Pinocchio.pdf)、[AMASS 官方说明](https://github.com/nghorbani/amass)。

## 本次结果（256 个 transition）

每类 64 个：直行、左转、右转、hold。检查各 transition 的 5 个内部时刻：

| 筛选阶段 | 通过数量 |
|---|---:|
| 几何兼容后的六维 nominal wrench 严格匹配 | 17 / 256 |
| Fx、Fy、Mz 硬匹配 + 接触/力矩约束，含时长搜索 | 63 / 256 |
| 再限制额外平衡补偿（20 N、5 Nm） | 33 / 256 |
| 所有内部帧通过 + 500 N/s 力变化率限制 | 22 / 256 |

最终为 **直行 2、左转 4、右转 1、零力 hold 15**。这是筛选后的候选集合，不是足够覆盖 locomotion 的训练集。逐帧通过样本的最大单手力 15.75 N、最大力变化率 335.04 N/s、最大关节力矩/额定力矩 0.499；独立 centroidal 动量平衡逐分量残差小于 8.1e-7（对应分量单位 N 或 Nm）。

例：右转 transition 97（原片段 yaw 变化 -18.83°，时长放大 2 倍）的一个中间时刻，世界系左手力约 `[-2.372,-0.035,0.007] N`，右手力约 `[3.497,1.266,-0.002] N`，合力 `[1.124,1.231,0.006] N`，COM 合力矩 `[-0.087,-0.403,-3.302] Nm`。两手不需要简单“同向拉”：差动力负责转向力偶。完整时间序列见 NPZ/CSV，不能把这个单帧数值当作整个 transition 的常值力。

报告：`summary.json`、`force_examples.csv`、`force_example.png`。4 项单元测试覆盖双手 rank-5 不可实现方向、旋转/平移坐标一致性、含地面的逆动力学与变化率不可行判定、匀速运动的非零引导信号，已通过。
