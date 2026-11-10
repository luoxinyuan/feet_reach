"""Plot diagnostic outputs without needing Isaac Sim."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());z=np.load(p/'traces.npz');t=np.arange(len(z['q']))*.02
fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for i,name in enumerate(['GT force prompt','Zero prompt','Oracle reference']):
 ax[0,0].plot(t,abs(z['q'][:,i]-z['target_q'][:,i]).mean(-1),label=name)
ax[0,0].set(ylabel='Joint MAE (rad)',xlabel='Time (s)');ax[0,0].legend()
j=r['joint_names'].index('right_shoulder_pitch_joint')
for name,data,style in [('Motion target',z['target_q'][:,0,j],'k--'),('Residual reference',z['reference_q'][:,0,j],'C1-'),('Actual robot',z['q'][:,0,j],'C0-')]:ax[0,1].plot(t,data,style,label=name)
ax[0,1].set(title='Right shoulder pitch',ylabel='Angle (rad)',xlabel='Time (s)');ax[0,1].legend()
ax[1,0].plot(t,np.linalg.norm(z['prompt'][:,0],axis=-1).sum(-1),label='Observed prompt');ax[1,0].plot(t,np.linalg.norm(z['physical_force'][:,0],axis=-1).sum(-1),label='Physical guidance');ax[1,0].set(ylabel='Sum of magnitudes (N)',xlabel='Time (s)');ax[1,0].legend()
for i,name in enumerate(['GT prompt','Zero prompt','Oracle reference']):ax[1,1].plot(t,abs(z['reference_q'][:,i]-z['target_q'][:,i]).mean(-1),label=name)
ax[1,1].set(ylabel='Reference-target MAE (rad)',xlabel='Time (s)');ax[1,1].legend()
for a in ax.flat:a.grid(alpha=.25)
fig.savefig(p/'diagnostic.png',dpi=150);plt.close(fig)
action=z['action'][1:,0];delta=np.diff(z['reference_q'][:,0],axis=0);active=abs(action)>.05;blocked=float(((abs(delta)<.1*.04*abs(action))&active).sum()/max(1,active.sum()));gap=abs(z['reference_q'][:,0]-z['q'][:,0]);near=float((gap>=.59).mean())
extra=dict(requested_updates_nearly_blocked_fraction=blocked,reference_gap_ge_0_59_fraction=near)
if 'matched_force_action_difference' in z:extra['matched_state_force_vs_zero_action_abs_difference']=float(abs(z['matched_force_action_difference']).mean())
(p/'diagnostics.json').write_text(json.dumps(extra,indent=2))
lines=['# Teacher residual 单轨迹诊断','',f'最终checkpoint；训练集clip {r["clip"]}，完整{r["duration_s"]:.2f}秒。根据目标动作选择：root平移<15cm、手臂位移包围盒对角线>40cm、长度400–800帧的首个clip；没有按policy表现挑选。', '', 'actor输入为policy＋当前111维force_gt_，无force history。checkpoint归一化逐项核对通过，force提示与motion索引逐帧一致。物理引导外力始终为零。', '', '|对照|实际q与目标MAE rad|累积参考与目标MAE rad|实际q与累积参考MAE rad|左右EE平均误差m|', '|---|---:|---:|---:|---|']
for c in r['cases']:lines.append(f"|{c['case']}|{c['joint_mae_rad']:.3f}|{c['reference_target_mae_rad']:.3f}|{c['lowlevel_reference_mae_rad']:.3f}|{c['ee_relative_error_m'][0]:.3f}, {c['ee_relative_error_m'][1]:.3f}|")
lines += ['', 'oracle_reference只用于诊断：把下一帧目标转成有原速度/限位约束的reference增量，仍经同一冻结CEER执行。它不属于训练好的policy，也不代表理想无误差底层。三组均不自动reset，完整跑完此clip，均未达到物理跌倒阈值。', '', f'实际提示平均总力 {r["prompt_mean_total_N"]:.3f} N。正常提示actor平均|tanh(action)|={r["cases"][0]["action_abs_mean"]:.3f}，并非零输出。约{blocked:.1%}的非微小增量请求实际reference更新小于请求量10%；{near:.1%}的时刻×关节样本，reference与实际q偏差≥0.59rad，接近原0.6rad保护边界。', '', '结论：此轨迹不是完全不动，但没有正确tracking。正常提示与零提示的总体跟踪误差接近；累积参考本身偏离目标，冻结底层对该参考也存在明显跟踪误差。不能据单个clip断言所有轨迹均如此，或确定是某个reward项单独导致；当前证据也不足以将问题归因于力编码本身。', '', '视频显示训练好的GT-prompt一组；其它两组只记录数值。青色为目标叠加（半透明mesh＋可见性辅助骨架），箭头仅显示最大三个提示力，不是实际外力；只显示env0机器人，避免多环境画面混淆。', '', '[完整视频](videos/clip00022_teacher_prompt.mp4)', '', '![诊断曲线](diagnostic.png)', '', '原始数据：traces.npz；result.json；diagnostics.json。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(extra,indent=2))
