"""Compare force budgets at identical gain and sample windows."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'artifacts/offline_force_baseline/20261105_calibrated_budget60'
s=json.loads((p/'summary.json').read_text())['metrics']
lines=['# 总力预算扩展到60 N', '', '与20 N基线保持相同的51段窗口、37点布局、固定K=2.5969108014710645。仅改变每帧Σ||Fi||₂预算。单点上限仍为30 N，平滑版单点变化率≤150 N/s。', '', '验证集24段、1536个transition。以下是Δq离线重建误差，不是policy tracking或成功率。', '', '|预算N|平滑|平均总力N|最大总力N|最大单点力N|RMSE rad|相对误差|', '|---:|---|---:|---:|---:|---:|---:|']
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for suffix in ['', '_smooth']:
 ms=[]
 for b in [20,30,40,60]:
  m=s[f'validation/offsets37/budget{b}{suffix}'];ms.append(m)
  lines.append(f"|{b}|{'是' if suffix else '否'}|{m['total_force_mean_N']:.3f}|{m['total_force_max_N']:.3f}|{m['point_max_N']:.3f}|{m['rmse_rad']:.6f}|{m['relative_error']:.2%}|")
 axs[0].plot([20,30,40,60],[m['relative_error']*100 for m in ms],'-o',label='Smoothed' if suffix else 'Budget only')
 axs[1].plot([20,30,40,60],[m['total_force_mean_N'] for m in ms],'-o',label='Smoothed' if suffix else 'Budget only')
axs[0].set(xlabel='Sum-of-magnitudes budget (N)',ylabel='Relative reconstruction error (%)');axs[1].set(xlabel='Sum-of-magnitudes budget (N)',ylabel='Mean sum of force magnitudes (N)')
for ax in axs:ax.grid(alpha=.3);ax.legend()
fig.savefig(p/'budget_comparison.png',dpi=160)
viol={};total=[];rmse=[]
for f in sorted((p/'clips').glob('*.json')):
 r=json.loads(f.read_text())
 for method,m in r['methods'].items():
  if 'solver' in m:
   for stage in ['stage1','stage2']:assert m['solver'][stage]['status'] in ['Solved','AlmostSolved']
   for k,v in m['solver']['violations'].items():viol[k]=max(viol.get(k,0),v)
 if r['record']['split']=='validation':
  z=np.load(f.with_suffix('.npz'));total.extend(np.linalg.norm(z['budget60_smooth'],axis=-1).sum(1).tolist());rmse.append((r['methods']['budget60_smooth']['relative_error'],r['record']['category'],f.stem))
lines += ['', '60 N平滑版逐帧总力分位数（验证集）：'+str(dict(zip(['p50','p90','p95','p99','max'],np.percentile(total,[50,90,95,99,100]).round(4).tolist()))), '', '全部解通过数值可行性检查；最大约束超差：'+str(viol), '', '该比较没有提高K，因此不会改善此前估算的力噪声敏感性。更大预算也不意味着每帧都使用60 N；第二阶段仍最小化用力。', '', '即使增加总预算，单点变化率约束仍可能阻止精确重建。固定姿态下的编码能力不等价于实际动力学可达性，也不能替代闭环policy评估。', '', '![预算比较](budget_comparison.png)', '']
(p/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
