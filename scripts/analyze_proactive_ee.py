"""Recompute transparent summary tables from a saved EE evaluation (no Isaac)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from eval_proactive_ee import summarize

def read_rows(path):
    with path.open() as f:
        return [{k:(v if k in ('phase','failure') else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);args=p.parse_args();out=args.experiment
    result=json.loads((out/'summary.json').read_text());conditions=sorted({t['condition'] for t in result['trials']})
    lookup={}
    for trial in result['trials']:
        name,r=trial['condition'],trial['repeat'];rows=read_rows(out/name/f'repeat_{r}'/'samples.csv');lookup[name,r]=rows
        trial.update(summarize(rows,trial['completed'],trial['failure'],result['config'].get('paired_reach_s',3.)))
        tc=trial['contact_time_s'];impact=[x for x in rows if tc is not None and tc<=x['time_s']<=tc+.1]
        trial['initial_100ms_peak_force_N']=max((x['normal_force_N'] for x in impact),default=None)
        trial['initial_100ms_peak_loading_N_s']=max((x['positive_force_rate_N_s'] for x in impact),default=None)
        trial['trial_duration_after_target_s']=max(x['time_s'] for x in rows)
        trial['loading_rate_over_500_fraction']=float(np.mean([x['positive_force_rate_N_s']>500. for x in rows]))
        hold=[x for x in rows if x['time_s']>=result['config'].get('paired_reach_s',3.)]
        trial['contact_hold_fraction']=float(np.mean([x['normal_force_N']>1. for x in hold])) if hold else None
        trial['max_feet_drift_m']=max(x['feet_drift_m'] for x in rows)
        trial['last_frame_failure']=rows[-1]['failure']
        trial['stop_reason']=None if trial.get('trajectory_finished',trial['completed']) else rows[-1]['failure']
        if hold and 'target_world_y' in hold[0]:
            trial['hold_tangential_rmse_m']=float(np.sqrt(np.mean([(x['hand_world_y']-x['target_world_y'])**2+(x['hand_world_z']-x['target_world_z'])**2 for x in hold])))
        (out/name/f'repeat_{r}'/'summary.json').write_text(json.dumps(trial,indent=2))
    comparisons=[]
    for r in range(result['config']['repeats']):
        wall=next(t for t in result['trials'] if t['condition']=='wall' and t['repeat']==r)
        tc=wall['contact_time_s'];comp=dict(repeat=r,wall_contact_s=tc)
        for name in conditions:
            base=json.loads((out/'no_wall'/f'repeat_{r}'/'start_state.json').read_text());start=json.loads((out/name/f'repeat_{r}'/'start_state.json').read_text())
            comp[name+'_start_root_delta_m']=float(np.linalg.norm(np.array(start['root_state'][:3])-base['root_state'][:3]))
            comp[name+'_start_joint_max_delta_rad']=float(np.max(abs(np.array(start['joint_pos'])-base['joint_pos'])))
            if 'policy_observation' in start:
                comp[name+'_start_policy_obs_max_delta']=float(np.max(abs(np.array(start['policy_observation'])-base['policy_observation'])))
        for name in ('wall','no_wall'):
            samples=[x for x in lookup[name,r] if tc is not None and max(0.,tc-.2)<=x['time_s']<tc-.02]
            valid=bool(samples) and not wall['contact_at_target_start']
            comp[name+'_precontact_samples']=len(samples)
            comp[name+'_mean_vx_m_s']=float(np.mean([x['vx_m_s'] for x in samples])) if valid else None
            comp[name+'_mean_positive_vx_m_s']=float(np.mean([max(0.,x['vx_m_s']) for x in samples])) if valid else None
        # Only compare wall/masked in a window preceding BOTH contact onsets.
        if 'wall_masked_geometry' in conditions:
            masked=next(t for t in result['trials'] if t['condition']=='wall_masked_geometry' and t['repeat']==r)
            times=[t for t in [tc,masked['contact_time_s']] if t is not None]
            end=min(times)-.02 if times else None
            for name in ('wall','wall_masked_geometry'):
                samples=[x for x in lookup[name,r] if end is not None and max(0.,end-.18)<=x['time_s']<end]
                valid=bool(samples) and not wall['contact_at_target_start'] and not masked['contact_at_target_start']
                comp[name+'_common_precontact_samples']=len(samples)
                comp[name+'_common_precontact_mean_positive_vx_m_s']=float(np.mean([max(0.,x['vx_m_s']) for x in samples])) if valid else None
        comparisons.append(comp)
    result['paired_precontact']=comparisons
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for filename,rows in [('metrics.csv',result['trials']),('paired_comparison.csv',comparisons)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    accuracy=result['accuracy'];completed=[a for a in accuracy if a.get('completed')]
    report=['# Student EE evaluation','',f"Checkpoint phase: `{result['config']['phase']}`; SHA256: `{result['config']['checkpoint_sha256']}`.",'',
        '## Free-space reaching','',f"Planned {len(accuracy)} targets; attempted {sum(a.get('attempted',False) for a in accuracy)}; completed {len(completed)}; tracking successes {sum(a.get('tracking_success',False) for a in accuracy)}.",
        'Accuracy below uses completed holds only; failures and unattempted targets remain in the planned-target denominator.','',
        '| Point | Completed | Mean hold error (mm) | P95 (mm) | Failure |','|---|---|---|---|---|']
    def fmt(x,scale=1.):return '—' if x is None else f'{x*scale:.2f}'
    for a in accuracy:report.append(f"| {a['point']} | {a.get('completed',False)} | {fmt(a.get('mean_hold_error_m'),1000)} | {fmt(a.get('p95_hold_error_m'),1000)} | {a.get('failure','')} |")
    report+=['',f"Mean of completed hold mean errors: {fmt(np.mean([a['mean_hold_error_m'] for a in completed]) if completed else None,1000)} mm.",'','[Accuracy video](accuracy/evaluation.mp4) · [Error curve](accuracy_errors.png)','','## Paired contact experiment','',f"Target root xyz: `{result['config']['paired_target_root']}`; wall x={result['config']['wall_x']} m.",result['config']['paired'],'',
        '| Condition | Completed | Contacted | Mean initial-impact peak (N, contacted only) | Failures |','|---|---|---|---|---|']
    for name in conditions:
        rows=[t for t in result['trials'] if t['condition']==name];contact=[t for t in rows if t['contact_time_s'] is not None]
        report.append(f"| {name} | {sum(t['completed'] for t in rows)}/{len(rows)} | {len(contact)}/{len(rows)} | {fmt(np.mean([t['initial_100ms_peak_force_N'] for t in contact]) if contact else None)} | {', '.join(sorted({t['failure'] for t in rows if t['failure']}))} |")
    report+=['','## Precontact measurements','',
        '| Seed repeat | Wall contact (s) | No-wall positive approach speed (m/s) | Wall positive approach speed (m/s) | Window samples (200 Hz) |','|---|---|---|---|---|']
    for c in comparisons:report.append(f"| {c['repeat']} | {fmt(c['wall_contact_s'])} | {fmt(c['no_wall_mean_positive_vx_m_s'])} | {fmt(c['wall_mean_positive_vx_m_s'])} | {c['wall_precontact_samples']} |")
    report+=['','Pointcloud masking is an OOD diagnostic, not a trained blind-policy baseline.','Precontact windows exclude the last 20 ms before detected >1 N contact; short windows must not be overinterpreted.','A trial with no contact is not a successful compliant contact. Steady force is unavailable when the hold was not reached.','Both raw initial states and proprioceptive observations are saved; see paired_comparison.csv for matching errors.','','[Contact/velocity curves](paired_curves.png) · [Full metrics](metrics.csv) · [Paired comparison](paired_comparison.csv)','','## Recordings','']
    for name in conditions:
        for r in range(result['config']['repeats']):report.append(f'- [{name}, seed {result["config"]["seed"]+r}]({name}/repeat_{r}/evaluation.mp4)')
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
if __name__=='__main__':main()
