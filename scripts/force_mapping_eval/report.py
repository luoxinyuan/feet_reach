import sys,json,shutil,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);r=json.loads((out/'evaluation/report.json').read_text());v=json.loads((out/'validation.json').read_text());m=json.loads((out/'manifest.json').read_text())
names=['jacobian3','jacobian9','force37'];labels=['3kp Jacobian','9kp Jacobian','37-point prompt'];colors=['#3984b8','#e59438','#338768']
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
a=axes[0];a.bar(labels,[r['summary'][n]['nrmse_mean'] for n in names],color=colors);a.axhline(r['summary']['state_only']['nrmse_mean'],ls='--',color='gray',label='State only');a.legend();a.set_ylabel('Normalized joint-delta RMSE (lower better)');a.set_ylim(0,1.05)
a=axes[1];means=np.array([r['summary'][n]['information_gain_mean'] for n in names]);ci=np.array([r['summary'][n]['information_gain_clip_bootstrap95'] for n in names]);a.bar(labels,means*100,color=colors);a.errorbar(labels,means*100,yerr=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means]))*100,fmt='none',color='black',capsize=4);a.axhline(0,color='gray');a.set_ylabel('SSE reduction vs state only (%)\n95% clip-bootstrap interval')
a=axes[2];a.bar(labels,[r['force_stats'][n]['total_mean_N'] for n in names],color=colors);a.set_ylabel('Mean sum of point-force norms (N)')
for a in axes:a.tick_params(axis='x',labelrotation=20);a.spines[['top','right']].set_visible(False)
fig.suptitle('Nominal reference-state audit — 113 held-out decoder test clips')
fig.savefig(out/'comparison.svg');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
lines=['# Force–motion 对应性评估（第一版）','', '这是离线名义姿态评估，不是 policy rollout，也不证明动态加权后的闭环效果。', '', '## 数据与协议','',f"560 个片段、82,363 个 transition；训练/验证/测试分别 363/84/113 个完整源片段。测试 16,328 帧。来自现有 AMASS+EE+静止混合库和 crouch 库，约 80/20。现有混合库经过旧 3kp 方法筛选，因此存在样本选择偏差。",'', '输入为当前 29 维关节位置、root-frame gravity 和 root height，加固定点位排列的力。没有速度、时间、motion ID、未来目标或 history。所有 decoder 同结构、同训练预算，3 个初始化种子；参数用验证集选择。所有方法上限相同：单点 30 N、各点范数和 60 N，但实际力分布不同。','', '主误差 NRMSE = sqrt(SSE / sum(Δq²))，越低越好；力信息增益 = 1 − SSE(state+force)/SSE(state)，越高越好。这不是百分比准确率。','', '|方法|NRMSE↓|相对状态 baseline 的 SSE 降低|95% clip-bootstrap CI|平均总力 N|', '|---|---:|---:|---:|---:|']
for n in ['state_only']+names:
 s=r['summary'][n];ci=s['information_gain_clip_bootstrap95'];force='—' if n=='state_only' else f"{r['force_stats'][n]['total_mean_N']:.2f}"
 lines.append(f"|{n}|{s['nrmse_mean']:.3f} ± {s['nrmse_seed_std']:.3f}|{s['information_gain_mean']*100:.1f}%|[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]|{force}|")
lines += ['', '## 简略结论','', '1. **37 点保存的关节动作信息最多**：全身 NRMSE 0.522，加入力后预测平方误差比 state-only 下降 67.8%。但它平均用了更大的总力（5.21 N），且可在腿部直接施力；这不是只改变施力点数量的消融。','2. **3kp 有明确增益，9kp 在当前 decoder 下增益不确定**：分别降低 21.3% 和 2.0%，后者置信区间跨零。9kp 是 9 个候选点中选 1–3 个，并不是同时施加 9 个力。不能据此声称 9kp policy 一定更差，或 9 点本身无效。','3. **37 点仍有明显学习/鲁棒性瓶颈**：解析 JᵀF/K 在相同测试集 NRMSE 为 %.3f，但统一小 MLP 为 0.522；因此 learned decoder 分数不等于编码本身的不可逆误差。0.1 N 各分量噪声（仅活跃点）让误差升至 0.643，说明需要考虑 estimator 噪声。' % v['force37_analytic_nrmse'], '4. **当前时序延迟不是 9kp 结果弱的主要解释**：使用平滑前、当前目标对应的 planned force，3kp 为 0.804，9kp 为 0.914；9kp 没有明显改善。', '', '## 分部位 NRMSE','', '|方法|左臂|右臂|腿|腰|','|---|---:|---:|---:|---:|']
for n in ['state_only']+names:lines.append('|'+n+'|'+'|'.join(f"{r['summary'][n]['groups'][g]:.3f}" for g in ['left_arm','right_arm','legs','waist'])+'|')
lines += ['', '37 点的腰部仍然恢复较弱，不能称全身动作已一一对应。', '', '## 其他检查','', '|方法|打乱力后的 NRMSE|活跃点 0.1 N 噪声后的 NRMSE|近似状态下按力选邻居：动作差异降低|','|---|---:|---:|---:|']
for n in names:lines.append(f"|{n}|{r['summary'][n]['shuffled_nrmse']:.3f}|{r['summary'][n]['noise_nrmse']:.3f}|{100*r['ambiguity'][n]['relative_gap_reduction']:.1f}%|")
lines += ['', '近邻指标与 decoder 不完全同序：9kp 的局部邻域差异降低 18.2%，略优于 3kp 的 15.4%。因此暂不将 9kp 的小 MLP 结果解释成信息不存在；可能涉及表示复杂度、采样随机性和数据量。近邻指标仅描述局部歧义，不是单射证明。', '', '## 已验证与局限','', f"- 数据有限、源片段划分无交叉、力上限通过；Jacobian 有限差分最大误差 {v['jacobian_finite_difference_max']:.2g}。", '- 解析解码仅作 37 点容量诊断，不用于三方法主排名，避免循环验证。', '- 3kp/9kp 在 reference pose 下复用生产求解/采样方法，按训练时四个物理 substep 重放 slew；这是运动学 replay，没有运行 PhysX policy。actor 可见的当前施力与 planned force 分别存储。', '- 整体速度匹配参考时反馈项为零；本次不评估实际-reference 偏离后的动态加权。', '- 37 点离线生成使用 64 帧块优化，具有未来轨迹信息；9kp/3kp 逐帧因果求解，生成器的信息条件不同。', '- 全部样本来自 policy 训练库，只对 decoder 做 held-out split；不是 policy OOD，也不是人物独立测试。', '- 3 个 decoder 种子固定；施力随机采样当前仅一组 seed。置信区间条件于这个数据划分，不涵盖生成器 seed/新人物变化。', '- 目前目标仅关节 Δq；root 平移/旋转尚未作为解码标签。', '- 图中总力是各点范数之和。JSON 的 slew 以旋转 root 坐标系计算，可能超过世界坐标的 150 N/s，不能据此判断原限幅失效。', '', '下一步建议：增加未经过 3kp 筛选的独立动作库和施力随机种子，做匹配实际总力的消融，再加入扰动状态与真实 rollout 的动态对应性测试。', '', '## 产物','', '- `evaluation/report.json`：完整指标，包含静止/运动、来源和各 seed。','- `evaluation/predictions.npz`、`*.pt`、`normalization.json`：预测、decoder 权重和预处理。','- `data.npz`、`manifest.json`：匹配数据与片段划分。','- `comparison.svg`：可编辑矢量图。','- `validation.json`：检查结果。','- 复现入口：`scripts/force_mapping_eval/README.md`。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
snapshot=out/'code';snapshot.mkdir(exist_ok=True)
for f in (root/'scripts/force_mapping_eval').glob('*'):
 if f.is_file():shutil.copyfile(f,snapshot/f.name)
for rel in ['active_adaptation/utils/jacobian_guidance.py','active_adaptation/utils/body_guidance.py','active_adaptation/utils/segmented_guidance.py','active_adaptation/envs/mdp/commands/jacobian_compliance.py']:
 f=root/rel;shutil.copyfile(f,snapshot/f.name)
(snapshot/'sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.iterdir() if f.is_file() and f.name!='sha256.json'},indent=2))
print(out/'REPORT.md')
