"""Summarize the saved deterministic validation runs and training logs."""
from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/passive_evaluation_20261020'
data={p.parent.name:json.loads(p.read_text()) for p in out.glob('*/summary.json') if '_video' not in p.parent.name}
fig,axes=plt.subplots(2,2,figsize=(11,7))
for row,task in enumerate(['compliance','locomotion']):
 labels=['teacher','adapt','student'];d=[data[f'{task}_{c}_guided'] for c in labels]
 axes[row,0].bar(labels,[100*x['completion_fraction'] for x in d],color=['#4977a8','#49a588','#dc8754'])
 axes[row,0].set_ylim(0,105);axes[row,0].set_ylabel('First-episode completion (%)');axes[row,0].set_title(task+' (completion is not tracking success)')
 key='ee_error_m' if task=='compliance' else 'root_xy_error_m'
 axes[row,1].bar(labels,[100*x['sample_weighted_metrics'][key] for x in d],color=['#4977a8','#49a588','#dc8754'])
 axes[row,1].set_ylabel('Time-weighted error (cm)');axes[row,1].set_title('Mean hand position error' if task=='compliance' else 'Root horizontal tracking error')
 for ax in axes[row]:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Held-out deterministic evaluation: 256 initializations per condition\nErrors measured until first failure / clip end; durations differ')
fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for row,task in enumerate(['compliance','locomotion']):
 run=Path((ROOT/f'outputs/passive-{task}/latest_pipeline.txt').read_text().strip())
 keys=['compliance/joint_mae_rad','compliance/ee_error_m'] if task=='compliance' else ['tracking/joint_mae_rad','tracking/root_xy_error_m']
 for phase in ['train','adapt','finetune']:
  s=(run/f'{phase}.log').read_text(errors='replace')
  for col,key in enumerate(keys):
   vals=[float(x) for x in re.findall(r'(?:^|\n)'+re.escape(key)+r': ([0-9.eE+-]+)',s)]
   axes[row,col].plot(np.linspace(0,1,len(vals)),vals,label=phase)
 for col,key in enumerate(keys):axes[row,col].set_title(key);axes[row,col].set_xlabel('Phase progress (logged segment)');axes[row,col].legend();axes[row,col].grid(alpha=.2)
fig.suptitle('Training diagnostics (stochastic rollouts; not validation metrics)')
fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
lines=['# Passive policy 评估（2026-10-20，America/New_York）','',
'结论：EE student 未达到最小验证要求；locomotion 已能在简单引导下移动并在零力站立，但跟踪不够准确，最终 finetune checkpoint 比 adaptation checkpoint 更差。','',
'## 方法','',
'从两个实验的最终 teacher / adaptation / finetune checkpoint 加载参数；冻结 observation normalization，使用确定性动作。每种条件 256 个初始化，seed=42。使用本地 val 数据；locomotion guided/zero 只采样运动片段，hold 只采样静止片段。teacher、adapt、student 的 guided 初始状态 SHA256 一致；student guided 与 zero 也一致。','',
'只统计第一次 episode，不用 reset 掩盖失败。到任务终止、片段结束、10秒上限或物理跌倒判据即停止该环境统计。跌倒判据为 root 高度 <0.45 m 或倾斜 >60°。未触发该判据仍可能弯腰、蹲下，不能视为姿态合格。completion 是没有提前失败并到达片段/时长终点，不是精确跟踪成功率。误差按有效时间加权，包含失败一步；不同策略存活时长不同，因此误差不能脱离完成率单独排序。256 个初始化有重复片段，不是256条独立动作。','',
'## 验证结果','',
'| 任务/策略/条件 | 完成比例 | 物理跌倒 | 平均观测时长(s) | 关节 MAE(rad) | Root XY误差(cm) | Yaw误差(°) | 双手平均误差(cm) |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for task in ['compliance','locomotion']:
 for controller,scenario in [('teacher','guided'),('adapt','guided'),('student','guided'),('student','zero'),('student','hold')]:
  d=data[f'{task}_{controller}_{scenario}'];m=d['sample_weighted_metrics'];ee=f"{100*m['ee_error_m']:.1f}" if 'ee_error_m' in m else '—'
  lines.append(f"| {task}/{controller}/{scenario} | {100*d['completion_fraction']:.1f}% | {100*d['physical_fall_fraction']:.1f}% | {d['mean_duration_s']:.2f} | {m['joint_mae_rad']:.3f} | {100*m['root_xy_error_m']:.1f} | {np.degrees(m['yaw_error_rad']):.1f} | {ee} |")
lines += ['',
'EE 的 guided 评估长达10秒，locomotion guided 平均约3.5秒（随机动作起点到clip末尾），两任务完成比例不可直接横比。locomotion hold 时长约5.98秒。','',
'## 解释','',
'- **EE：教师能站住，但手部跟踪本身也不好。** 教师 guided 完成率96.1%，平均双手目标误差18.2cm；student 为39.5%、22.5cm。student 双手都在目标5cm内的有效时间仅1.13%。视频中存在明显弯腰/姿态塌陷。','- **EE 的主要稳定性退化出现在 adaptation。** adaptation guided 256/256触发物理跌倒，平均1.52秒；最终student稍有恢复，但零力hold仍256/256跌倒，平均1.29秒。作为对照，teacher零力hold完成255/256。不能声称已经学会 passive compliance。','- **Locomotion 有初步效果。** 最终student guided完成229/256，物理跌倒6/256，另有任务提前终止；root水平误差11.5cm，yaw误差9.5°。零力站立完成256/256，平均root水平偏移3.7cm。','- **Locomotion finetune 没有带来提升。** adaptation完成率91.0%，root误差6.2cm、yaw误差5.6°；最终student对应89.5%、11.5cm、9.5°。建议暂把adapt checkpoint作为后续比较基线。','- **引导信号有作用，但尚未证明一般的人机交互控制。** 从相同运动初始状态移除引导，locomotion片段完成率降至24.6%，大多是参考跟踪终止（不是摔倒）。这一对照不等于“行走中撤力后能立即停住”的测试。','',
'## 建议的下一步','',
'先修复EE零力站立/蒸馏稳定性，再扩展手部transition；结合teacher动作监督检查latent蒸馏误差及分布偏移，不能仅靠增加训练步数。Locomotion优先保留adapt模型，缩小finetune更新并按验证集保存最佳checkpoint，然后补行走中撤力、持续转向和未见过的手动引导测试。这些是根据评估提出的下一步，并未修改或重启训练。','',
'## 文件','',
'- [评估图](evaluation.png)、[训练曲线](training_curves.png)。',
'- [EE guided 视频](compliance_student_guided_video/rollout.mp4)、[EE 零力视频](compliance_student_hold_video/rollout.mp4)。',
'- [Locomotion guided 视频](locomotion_student_guided_video/rollout.mp4)、[Locomotion 零力站立视频](locomotion_student_hold_video/rollout.mp4)。',
'- 每个条件子目录的summary.json含checkpoint路径、初始状态hash及指标，traces.npz保存前8个环境的轨迹；视频是单独单环境演示，不用于上述256环境统计。',
'- 可复现脚本：`scripts/eval_passive_policies.py`、`scripts/analyze_passive_evaluation.py`。','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(out/'REPORT.md')
