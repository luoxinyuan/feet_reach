"""Compare fixed-target reports; match completed trials before comparing accuracy."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


def percentile(values, fraction):
    values=sorted(values)
    if not values: return None
    index=(len(values)-1)*fraction
    lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('v1',type=Path);p.add_argument('v2',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    dirs=[args.v1,args.v2]
    configs=[json.loads((d/'config.json').read_text()) for d in dirs]
    for field in ('continue_on_instability','seed','repeats','protocol','settle_s','reach_s','hold_s','coordinate_frame','failure','external_force_enabled','rng_seed_timing'):
        assert configs[0][field]==configs[1][field],f'Protocol mismatch: {field}'
    targets=[json.loads((d/'targets.json').read_text()) for d in dirs]
    assert targets[0]==targets[1], 'Target positions differ'
    reports=[json.loads((d/'summary.json').read_text()) for d in dirs]
    trials=[{(r['point'],r['repeat']):r for r in s['trials']} for s in reports]
    expected={(i,j) for i in range(len(targets[0])) for j in range(configs[0]['repeats'])}
    assert all(set(t)<=expected for t in trials), 'Unexpected trial IDs'
    common={key for key in expected if all(key in t and t[key]['completed_hold'] for t in trials)}
    all_samples=[]
    for d in dirs:
        with (d/'samples.csv').open() as f:
            all_samples.append(list(csv.DictReader(f)))
    results=[]
    for t,rows in zip(trials,all_samples):
        errors=[float(r['error_m'])*1000 for r in rows if r['phase']=='hold' and (int(r['point']),int(r['repeat'])) in common]
        completed=[r['mean_error_m']*1000 for r in t.values() if r['completed_hold']]
        results.append(dict(planned_targets=len(expected), unattempted_targets=len(expected)-len(t), trials=len(t),successes=sum(r['success'] for r in t.values()),
            failures=sum(r['failed'] for r in t.values()),failure_reasons=dict(Counter(r['failure'] for r in t.values() if r['failed'])),
            completed_trials=sum(r['completed_hold'] for r in t.values()),
            completed_only_mean_mm=sum(completed)/len(completed) if completed else None,
            common_completed_trials=len(common),common_hold_frames=len(errors),
            matched_mean_mm=sum(errors)/len(errors) if errors else None,
            matched_rmse_mm=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            matched_p95_mm=percentile(errors,.95), matched_max_mm=max(errors) if errors else None))
    out=dict(protocol=configs[0],v1=results[0],v2=results[1],
             checkpoints=[c['checkpoint'] for c in configs],matched_trial_ids=sorted(common),
             note='No external force at evaluation. Accuracy matched on trials with completed holds in both policies; failures remain in full success-rate denominator. Continuous sampled-target benchmark, not full-workspace robustness.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    def fmt(v):return 'N/A' if v is None else f'{v:.2f}'
    a,b=results
    lines=['# v1 与 v2 固定脚目标评测对比','',
           f"同一程序、seed={configs[0]['seed']}、{len(targets[0])} 个目标 × {configs[0]['repeats']} 次；连续发送，中途不重置，无外力。仅起始稳定 {configs[0]['settle_s']} s → 直接切换目标并 reaching {configs[0]['reach_s']} s → 保持 {configs[0]['hold_s']} s。",
           "成功标准：全过程不触发失稳并完成保持；位置误差不参与成功判定。",'',
           '| 指标 | v1 | v2 |','|---|---:|---:|',
           f"| 成功次数 | {a['successes']}/{a['trials']} | {b['successes']}/{b['trials']} |",
           f"| 未尝试目标数 | {a['unattempted_targets']} | {b['unattempted_targets']} |",
           f"| 失稳次数 | {a['failures']} | {b['failures']} |",
           f"| 完成保持次数 | {a['completed_trials']} | {b['completed_trials']} |",
           f"| 各自完成试验的平均误差（mm，样本可能不同） | {fmt(a['completed_only_mean_mm'])} | {fmt(b['completed_only_mean_mm'])} |",
           f"| 共同完成 {len(common)} 次试验：平均误差（mm） | {fmt(a['matched_mean_mm'])} | {fmt(b['matched_mean_mm'])} |",
           f"| 共同完成试验：RMSE（mm） | {fmt(a['matched_rmse_mm'])} | {fmt(b['matched_rmse_mm'])} |",
           f"| 共同完成试验：P95（mm） | {fmt(a['matched_p95_mm'])} | {fmt(b['matched_p95_mm'])} |",'',
           '| 点编号 | root 系 XYZ（m） | v1 成功 | v2 成功 |','|---|---|---:|---:|']
    for i,target in enumerate(targets[0]):
        counts=[sum(r['success'] for key,r in t.items() if key[0]==i) for t in trials]
        lines.append(f"| {i} | {', '.join(f'{x:.3f}' for x in target)} | {counts[0]}/{configs[0]['repeats']} | {counts[1]}/{configs[0]['repeats']} |")
    lines+=['',f"v1 失稳原因：{a['failure_reasons']}；v2：{b['failure_reasons']}。",'',
            '共同完成试验的精度比较排除了两者任一失败的试验，必须结合全体成功率理解，不能据此忽略失败。',
            '本轮未施加测试外力，不能据此断言抗 0–20 N 外力性能；也不能推断整个工作空间的成功率。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
