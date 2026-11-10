"""Aggregate per-trial metrics without hiding failed or short episodes."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(mean=float(x.mean()),p90=float(np.percentile(x,90)),count=int(x.size)) if x.size else dict(mean=None,p90=None,count=0)

def format_metric(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'

def tracking(path):
    r=json.loads((path/'result.json').read_text());d=dict(np.load(path/'traces.npz'));out=[]
    keys=[k for k in d if k.endswith(('_mae_rad','_error_m','_error_rad'))]+['force_sum_N','force_peak_N','force_slew_N_s','active_points']
    for i,row in enumerate(r['rows']):
        mask=d['valid'][:,i];fall=int(d['first_fall_step'][i]);expected=row['frames']-1
        record=dict(row,seed=r['seed'],physical_completed=fall<0 and r['steps']>=expected,fall_time_s=fall*r['dt'] if fall>=0 else None,observed_seconds=min(int(mask.sum())-1,expected)*r['dt'])
        for k in keys:record[k]=stats(d[k][:,i][mask])
        for side in ['left_arm','right_arm']:
            m=mask & (d[side+'_stationary'][:,i]>.5);m[0]=False
            record[side+'_stationary_motion_rad_s']=stats(d[side+'_stationary_motion_rad_s'][:,i][m])
        # Explicit tracking criterion, independent of each checkpoint's training termination.
        m=mask & (d['root_xy_error_m'][:,i]<.5) & (d['joint_mae_rad'][:,i]<.5)
        record['within_tracking_tolerance_fraction']=float(m.sum()/max(min(row['frames'],len(mask)),1))
        out.append(record)
    return out

def behavior(path):
    r=json.loads((path/'result.json').read_text());forced=dict(np.load(path/'traces.npz'));zero=dict(np.load(path.parent/'behavior_zero/traces.npz'));n=len(r['rows']);dt=r['dt'];out=[]
    zr=json.loads((path.parent/'behavior_zero/result.json').read_text())
    assert r['rows']==zr['rows'] and r['seed']==zr['seed']
    assert r['checkpoint_sha256']==zr['checkpoint_sha256'] and r['manifest_sha256']==zr['manifest_sha256']
    for key in ['initial_joint_pos','initial_root']:
        np.testing.assert_allclose(forced[key],zero[key],atol=1e-6,rtol=1e-6)
    mismatch=float(np.max(np.abs(forced['root'][:101]-zero['root'][:101])))
    assert mismatch<.002, f'Pre-force paired mismatch: {mismatch}'
    (path.parent/'paired_validation.json').write_text(json.dumps(dict(preforce_max_root_difference_m=mismatch),indent=2))
    d={k:np.concatenate([v,zero[k]],axis=0 if k.startswith('initial_') or k=='first_fall_step' else 1) for k,v in forced.items()}
    yaw=np.unwrap(d['yaw'],axis=0);qs=d['torso_relative_quat'];angles=Rotation.from_quat(qs.reshape(-1,4)[:,[1,2,3,0]]).as_euler('xyz').reshape(*qs.shape[:2],3)
    angles=np.unwrap(angles,axis=0)
    start=round(2/dt);end=round(8/dt);last=len(yaw)-1
    for i,row in enumerate(r['rows']):
        z=i+n;fall=int(d['first_fall_step'][i]);zfall=int(d['first_fall_step'][z]);safe=fall<0;paired_safe=safe and zfall<0
        def corrected(x):return (x[:,i]-x[start,i])-(x[:,z]-x[start,z])
        root=corrected(d['root']);hands=corrected(d['hands_b']);yd=corrected(yaw);ang=corrected(angles)
        group=row['group'];unit='m'
        if group=='ee':side=0 if row['side']=='left' else 1;response=hands[:,side]@np.array(row['direction']);leak=float(np.linalg.norm(hands[end,1-side]))
        elif group in ('locomotion','crouch'):response=root@np.array(row['direction']);leak=float(abs(np.rad2deg(yd[end])))
        elif group=='yaw':response=np.rad2deg(yd)*row['sign'];leak=float(np.linalg.norm(root[end,:2]));unit='deg'
        else:response=np.rad2deg(ang[:,row['axis']])*row['sign'];leak=float(np.linalg.norm(root[end,:2]));unit='deg'
        threshold=.02 if unit=='m' else 3.
        # Response time requires a sustained 0.1 s threshold crossing.
        hits=response[start:end+1]>threshold;window=max(1,round(.1/dt));hit=np.flatnonzero(np.convolve(hits.astype(int),np.ones(window,dtype=int),'valid')>=window)
        response_end=float(response[end]) if paired_safe else None
        bodies=[b for b,_ in row['loads']];visible=all(b in r['force_input_bodies'] for b in bodies)
        kneeids=[r['joint_names'].index(s+'_knee_joint') for s in ('left','right')]
        knee=corrected(d['joint_pos'][:,:,kneeids])
        out.append(dict(row,force_input_supported=visible,physical_survived=safe,paired_survived=paired_safe,
            fall_time_s=fall*dt if fall>=0 else None,response_unit=unit,response_at_release=response_end,
            response_per_N=response_end/row['force_N'] if response_end is not None else None,
            response_latency_s=float(hit[0]*dt) if len(hit) and paired_safe else None,
            non_target_response=leak if paired_safe else None,
            corrected_knee_flexion_rad=float(knee[end].mean()) if paired_safe else None,
            release_root_drift_m=float(np.linalg.norm(d['root'][last,i]-d['root'][end,i])) if safe else None,
            release_yaw_drift_deg=float(np.rad2deg(yaw[last,i]-yaw[end,i])) if safe else None,
            final_root_speed_m_s=float(np.linalg.norm(d['root_velocity'][-50:,i],axis=-1).mean()) if safe else None,
            zero_root_drift_m=float(np.linalg.norm(d['root'][last,z]-d['root'][0,z])) if zfall<0 else None,
            zero_yaw_drift_deg=float(np.rad2deg(yaw[last,z]-yaw[0,z])) if zfall<0 else None,
            directed_response_success=bool(paired_safe and response[end]>(.05 if unit=='m' else 10))))
    return out

def build(root):
    root=Path(root);data=root/'metrics' if (root/'metrics').exists() else root;tr=[];br=[];runs=[]
    for p in sorted(data.glob('tracking_seed*')):
        if (p/'result.json').exists():tr+=tracking(p);runs.append(json.loads((p/'result.json').read_text()))
    if (data/'behavior/result.json').exists() and (data/'behavior_zero/result.json').exists():br=behavior(data/'behavior');runs.append(json.loads((data/'behavior/result.json').read_text()))
    result=dict(tracking_trials=tr,behavior_trials=br)
    result['tracking_summary']={}
    for split in sorted(set(x['split'] for x in tr)):
        for cat in ['all']+sorted(set(x['category'] for x in tr if x['split']==split)):
            rows=[x for x in tr if x['split']==split and (cat=='all' or x['category']==cat)]
            result['tracking_summary'][split+'/'+cat]=dict(trials=len(rows),physical_completion_rate=float(np.mean([x['physical_completed'] for x in rows])),
                 **{k:stats([x[k]['mean'] for x in rows if x[k]['mean'] is not None]) for k in ('joint_mae_rad','root_xy_error_m','root_z_error_m','yaw_error_rad','ee_world_error_m','ee_relative_error_m')})
    result['behavior_summary']={}
    for group in sorted(set(x['group'] for x in br)):
        for force in [5,10,20]:
            rows=[x for x in br if x['group']==group and x['force_N']==force]
            result['behavior_summary'][f'{group}/{force}N']=dict(trials=len(rows),survival=float(np.mean([x['physical_survived'] for x in rows])),directed_response_rate=float(np.mean([x['directed_response_success'] for x in rows])),
                input_supported=all(x['force_input_supported'] for x in rows),response_unit=rows[0]['response_unit'],response=stats([x['response_at_release'] for x in rows if x['response_at_release'] is not None]))
    result['zero_force_summary']={}
    if (data/'behavior_zero/result.json').exists():
        zr=json.loads((data/'behavior_zero/result.json').read_text());runs.append(zr)
        zd=dict(np.load(data/'behavior_zero/traces.npz'));safe=zd['first_fall_step']<0
        result['zero_force_summary']=dict(trials=int(len(safe)),survival=float(safe.mean()),
            root_drift_m=stats(np.linalg.norm(zd['root'][-1]-zd['root'][0],axis=-1)[safe]),
            yaw_drift_abs_deg=stats(np.abs(np.rad2deg(np.unwrap(zd['yaw'],axis=0)[-1]-zd['yaw'][0]))[safe]),
            final_root_speed_m_s=stats(np.linalg.norm(zd['root_velocity'][-50:],axis=-1).mean(0)[safe]))
    result['serial_worker_seconds']=sum(x['wall_seconds'] for x in runs)
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    prompt_only=any(r.get('prompt_only_behavior',False) for r in runs)
    behavior_protocol=('Behavior supplies force tokens to the trained actor interface; no physical wrench is dispatched.' if prompt_only else 'Behavior uses identical physical loads.')
    lines=['# Compliance evaluation '+(runs[0]['version'] if runs else 'unknown'),'', 'Tracking uses each checkpoint’s own guidance generator. '+behavior_protocol+' No combined score.', '', '## Tracking','', '| Split/category | Trials | Physical completion | Joint MAE (rad) | EE relative (m) | Root XY (m) |','|---|---:|---:|---:|---:|---:|']
    for name,s in result['tracking_summary'].items():lines.append(f"| {name} | {s['trials']} | {s['physical_completion_rate']:.1%} | {format_metric(s['joint_mae_rad']['mean'])} | {format_metric(s['ee_relative_error_m']['mean'])} | {format_metric(s['root_xy_error_m']['mean'])} |")
    behavior_heading='## Force-token prompts (no physical wrench)' if prompt_only else '## Fixed forces'
    lines+=['',behavior_heading,'','| Group/force | Trials | Survival | Directed response | Force points in actor interface |','|---|---:|---:|---:|---|']
    for name,s in result['behavior_summary'].items():lines.append(f"| {name} | {s['trials']} | {s['survival']:.1%} | {s['directed_response_rate']:.1%} | {s['input_supported']} |")
    if prompt_only:
        lines.append('')
        lines.append('Prompt-only support is checked by exact body and local offset. Unsupported behavior locations receive no force token and no physical wrench, so their zero response is not evidence about force-prompt decoding.')
    if result['zero_force_summary']:
        z=result['zero_force_summary']
        lines+=['','## Zero force','',f"12 s survival: {z['survival']:.1%}; mean root drift: {format_metric(z['root_drift_m']['mean'])} m; mean absolute yaw drift: {format_metric(z['yaw_drift_abs_deg']['mean'], 1)} degrees."]
    lines+=['','Directed response is survival plus >5 cm or >10 degrees at release, relative to a paired zero-force trial. It is a coarse diagnostic, not a claim that larger responses are better. Failed trials have no end-response score; survival is always reported separately.',
            'Physical fall: root height <0.30 m or root tilt >75 degrees. Tracking completion means survival through the requested clip, not tracking success. Training-specific cumulative-error termination is disabled.',
            'Validation denotes the stored dataset split; original AMASS recording-level disjointness cannot be certified. Train diagnostics are checkpoint-specific exposure labels; re-audit for other checkpoints.',
            'Numerical category labels are motion-feature heuristics, not human action annotations. Empty categories must be reported, not silently replaced.','',f"Total serial worker time: {result['serial_worker_seconds']/60:.1f} min."]
    prompt_audits=[]
    for p in sorted(data.glob('video_tracking_*')):
        if not (p/'traces.npz').exists() or not (p/'result.json').exists():continue
        with np.load(p/'traces.npz') as d:
            if 'actor_prompt_input_sum_N' not in d or 'reference_root' not in d:continue
            r=json.loads((p/'result.json').read_text());row=r['rows'][0]
            if 'force_gt_' not in r.get('actor_inputs',[]):continue
            prompt=d['actor_prompt_input_sum_N'][:,0];actual=d['root'][:,0];reference=d['reference_root'][:,0];error=d['root_xy_error_m'][:,0]
            prompt_audits.append((p.name,row,int((prompt>1e-4).sum()),len(prompt),float(prompt.max()),float(np.linalg.norm(reference[-1]-reference[0])),float(np.linalg.norm(actual[-1]-actual[0])),float(error.mean()),float(error[-1])))
    if prompt_audits:
        lines+=['','## Actor force-token and motion audit','','| Clip | Category | Nonzero-token frames | Peak token sum (N) | Target root displacement (m) | Actual root displacement (m) | Root XY error mean/end (m) |','|---|---|---:|---:|---:|---:|---:|']
        for name,row,active,total,peak,target_move,actual_move,error_mean,error_end in prompt_audits:
            lines.append(f"| {name} | {row['category']} | {active}/{total} | {peak:.2f} | {target_move:.2f} | {actual_move:.2f} | {error_mean:.2f}/{error_end:.2f} |")
    lines+=['','## Representative videos','']
    videos=sorted((root/'videos').glob('*.mp4')) if (root/'videos').exists() else sorted(root.glob('video*/rollout.mp4'))
    for video in videos:
        meta_path=(data/('video_'+video.stem)/'result.json') if video.parent.name=='videos' else video.parent/'result.json'
        if not meta_path.exists():continue
        meta=json.loads(meta_path.read_text())
        lines.append(f"- [{video.stem}]({video.relative_to(root)}): {meta['steps']*meta['dt']:.1f} s, {video.stat().st_size/1e6:.2f} MB")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();build(a.root)
