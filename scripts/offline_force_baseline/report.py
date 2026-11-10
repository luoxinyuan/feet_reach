"""Render the completed offline benchmark (no simulator required)."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/offline_force_baseline'
a=json.loads((BASE/'20261105_baseline_K80/summary.json').read_text())['metrics']
b=json.loads((BASE/'20261105_calibrated/summary.json').read_text())['metrics']
out=BASE/'report';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for layout in ['origins23','origins29','offsets37']:
 rows=[a[f'validation/{layout}/budget{x}_smooth'] for x in [10,20,40,60]]
 axes[0].plot([10,20,40,60],[r['relative_error'] for r in rows],'-o',label=layout)
axes[0].set(title='Fixed K=80, validation',xlabel='Sum-of-magnitudes budget (N)',ylabel='Relative joint-increment error');axes[0].legend()
for suffix in ['', '_smooth']:
 rows=[b[f'validation/offsets37/budget{x}{suffix}'] for x in [5,10,20]]
 axes[1].plot([5,10,20],[r['relative_error'] for r in rows],'-o',label='Smoothed' if suffix else 'Budget only')
axes[1].set(title='Train-calibrated K=2.59691, 37 points',xlabel='Sum-of-magnitudes budget (N)',ylabel='Relative joint-increment error');axes[1].legend()
for ax in axes:ax.grid(alpha=.3)
fig.savefig(out/'budget_comparison.png',dpi=160);fig.savefig(out/'budget_comparison.pdf');plt.close(fig)
rows=[]
for path in (BASE/'20261105_calibrated/clips').glob('*.json'):
 r=json.loads(path.read_text())
 if r['record']['split']=='validation':rows.append((path,r))
path,r=max(rows,key=lambda z:z[1]['methods']['budget20_smooth']['total_force_mean_N'])
z=np.load(path.with_suffix('.npz'));f=z['budget20_smooth'];t=np.arange(len(f))*.02
fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
axes[0].plot(t,np.linalg.norm(f,axis=-1).sum(1),label='Sum of magnitudes');axes[0].plot(t,np.linalg.norm(f.sum(1),axis=-1),label='Magnitude of vector sum');axes[0].axhline(20,color='r',ls='--',label='Budget');axes[0].set_ylabel('Force (N)');axes[0].legend()
axes[1].plot(t[1:],np.linalg.norm(np.diff(f,axis=0),axis=-1).max(1)/.02);axes[1].axhline(150,color='r',ls='--');axes[1].set(xlabel='Time (s)',ylabel='Max point slew (N/s)')
fig.savefig(out/'force_sequence.png',dpi=160);plt.close(fig)
lines=['# 离线全身力编码基线', '', '本次只完成离线计算和编码评估，未启动 policy 训练。不是动力学 rollout，也不是运动跟踪成功率评估。', '', '## 定义', '', '`A=[J_1,q^T ... J_m,q^T]`，`tau_prompt=K Δq`，`Δq_hat=A F/K`。仅编码29个驱动关节增量，不编码浮动基座平移/转动。', '', '每帧严格约束 **Σ_i ||F_i||₂ ≤ B**；反向力不能相互抵消预算。单点上限30 N。平滑版还约束每点变化率≤150 N/s（50 Hz下≤3 N/帧）。不加入root合力或合力矩匹配。', '', '两阶段SOCP：先在上述约束下最小化关节增量重建误差，再允许RMSE增加至多0.0001 rad，最小化平均总力并加入时间平滑项。没有求解后裁剪。', '', '## 样本和可表达性', '', '固定manifest中27个训练诊断窗口、24个验证窗口，每段64个连续transition，共3264个transition。覆盖手臂、移动、转向、弯腰、下蹲等类别；并非完整AMASS预计算。原始录制级训练/验证独立性未核实。', '', '| 候选布局 | 样本中Jacobian秩 | 说明 |', '|---|---:|---|', '|23个link原点|21/29|末端关节缺少力臂|', '|29个关节子link原点|25/29|仍缺少末端旋转表达|', '|37点：29原点+8个偏移点|29/29|手和脚各增加两个非共线5cm偏移点|', '', '**37点属于虚拟几何诊断布局，尚未确认都位于可接触表面。** 不限制点数不能自动解决秩不足，施力位置也必须有合适力臂。', '', '## 验证结果', '', '所有误差均针对逐帧Δq重建，不是对policy轨迹的评价。relative error为全样本误差范数/目标范数。', '', '|K|布局|总力预算N|平均总力N|实际最大总力N|Δq RMSE rad|相对误差|', '|---:|---|---:|---:|---:|---:|---:|']
for gain,data,layout,budgets in [(80,a,'offsets37',[20,60]),(2.59691,b,'offsets37',[5,10,20])]:
 for budget in budgets:
  m=data[f'validation/{layout}/budget{budget}_smooth'];lines.append(f"|{gain:g}|{layout}|{budget}|{m['total_force_mean_N']:.3f}|{m['total_force_max_N']:.3f}|{m['rmse_rad']:.6f}|{m['relative_error']:.1%}|")
noise=np.mean([r['force_noise_0_1N_decode_rmse_rad'] for _,r in rows])
lines += ['', 'K=2.59691由训练窗口的无约束解99分位力需求标定，验证集未用于调K。K在所有帧固定；更小K本身不是更好的优化算法。', '', f'若每个力分量存在独立0.1 N标准差噪声，验证窗口平均理论解码RMS约 **{noise:.4f} rad**。这是诊断噪声假设，不是已测量estimator误差；可能超过原始Δq信号，因此不能据无噪声低误差宣称训练一定成功。', '', '建议把 **37点、K=2.59691、总预算20 N、变化率150 N/s** 作为低力离线对照配置。它牺牲少量编码精度来减小外力；下一步需要真实force estimator噪声与闭环policy验证。', '', '## 验证与产物', '', '单元测试：反向力预算、时间变化率约束、偏移点Jacobian有限差分，3项通过。所有输出检查总力/单点预算和变化率可行性。高增益对照一例数值求解失败，使用相同问题、较宽数值收敛容差重试，仍执行相同显式可行性检查。', '', '完整结果：`../20261105_baseline_K80/` 和 `../20261105_calibrated/`，每段NPZ含q、Δq、施力点位置和各方案完整力序列；manifest记录源clip与帧号。初末帧力未强制为零，部署须另处理启动/结束过渡。', '', f'连续力示例来自验证窗口 `{path.stem}`，类别 `{r["record"]["category"]}`；选取平均总力最高的验证窗口，不是最好看窗口。', '', '![预算比较](budget_comparison.png)', '', '![连续力序列](force_sequence.png)', '']
(out/'REPORT.md').write_text('\n'.join(lines))
print(out/'REPORT.md')
