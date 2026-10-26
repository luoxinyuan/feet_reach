"""Matched held-out/interaction evaluation; explicitly separate task metrics."""
import json,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('artifacts/ceer_segmented_eval_20261024'))
parser.add_argument('--baseline-tracking',type=Path,default=Path('artifacts/ceer_force_eval_20261024/jacobian9/summary.json'))
parser.add_argument('--baseline-force-suite',type=Path)
parser.add_argument('--baseline-label',default='old_jacobian9')
parser.add_argument('--label',default='segmented18')
a=parser.parse_args();root=a.root
baseline_suite=a.baseline_force_suite or root/'old_force_suite'
oldtracking=json.loads(a.baseline_tracking.read_text())
newtracking=json.loads((root/'guided/summary.json').read_text())
assert oldtracking['initial_sha256']==newtracking['initial_sha256']
assert oldtracking['reference_sha256']==newtracking['reference_sha256']
result={'tracking_matched_initial_and_reference':True,'tracking':{},'interaction':{}}
for name,r in [(a.baseline_label,oldtracking),(a.label,newtracking)]:
 m=r['sample_weighted_metrics'];result['tracking'][name]=dict(completion=r['completion_fraction'],physical_fall=r['physical_fall_fraction'],joint_mae=m['joint_mae_rad'],keypoint_cm=m['keypoint_error_m']*100,root_xy_cm=m['root_xy_error_m']*100,yaw_deg=np.degrees(m['yaw_error_rad']),mean_force_N=m['force_N'],valid_duration_s=r['mean_duration_s'])
for name,path in [(a.baseline_label,baseline_suite),(a.label,root/'force_suite')]:
 r=json.loads((path/'comparison.json').read_text());f=dict(np.load(path/'metrics/forced.npz'));z=dict(np.load(path/'metrics/zero.npz'));rows=json.loads((path/'metrics/pass_forced.json').read_text())['rows']
 stats={'zero':r['zero_control'],'preprompt_max_difference':r['preprompt_max_root_difference_m'],'by_force':{},'hand_response':{}}
 for force in [5,10,20]:
  records=[x for x in r['records'] if x['force_N']==force]
  stats['by_force'][force]=dict(n=len(records),stable=sum(x['stable'] for x in records),directional_response=sum(x['success'] for x in records))
  # Retain all trials here; also report a separate matched-survivor analysis.
  contralateral=[];ipsilateral=[];hand_contra=[];hand_ipsi=[];n_survive=0
  for i,row in enumerate(rows):
   if row['force_N']!=force or row['group'] not in ('left','right'):continue
   side=row['group'];other='right' if side=='left' else 'left';qnames=f['joint_names'].tolist()
   def arm_ids(s):return [j for j,n in enumerate(qnames) if n.startswith(s+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
   # Baseline-subtracted change removes spontaneous motion in the zero run.
   delta=(f['joint_pos'][100:301,i]-f['joint_pos'][100,i])-(z['joint_pos'][100:301,i]-z['joint_pos'][100,i])
   contralateral.append(float(np.abs(delta[:,arm_ids(other)]).mean()));ipsilateral.append(float(np.abs(delta[:,arm_ids(side)]).mean()))
   dh=(f['hands_b'][100:301,i]-f['hands_b'][100,i])-(z['hands_b'][100:301,i]-z['hands_b'][100,i])
   hand_contra.append(float(np.linalg.norm(dh[:,0 if other=='left' else 1],axis=-1).mean()));hand_ipsi.append(float(np.linalg.norm(dh[:,0 if side=='left' else 1],axis=-1).mean()))
   n_survive+=int(not np.isfinite(f['first_fall_s'][i]))
  stats['hand_response'][force]=dict(n=len(contralateral),stable=n_survive,unforced_arm_q_mae_rad=float(np.mean(contralateral)),forced_arm_q_mae_rad=float(np.mean(ipsilateral)),unforced_hand_local_m=float(np.mean(hand_contra)),forced_hand_local_m=float(np.mean(hand_ipsi)),unforced_over_forced_q_ratio=float(np.mean(contralateral)/max(np.mean(ipsilateral),1e-9)))
 result['interaction'][name]=stats
# Compare only the SAME trials that survive under BOTH policies; report this
# explicitly as a survivor-conditioned diagnostic, not an overall success metric.
old=json.loads((baseline_suite/'comparison.json').read_text())
new=json.loads((root/'force_suite/comparison.json').read_text())
assert [ (r['name'],r['force_N'],r['repeat']) for r in old['records'] ]==[ (r['name'],r['force_N'],r['repeat']) for r in new['records'] ]
result['matched_survivor_hand_response']={}
for force in [5,10]:
 ids=[i for i,(o,n) in enumerate(zip(old['records'],new['records'])) if o['force_N']==force and o['group'] in ('left','right') and o['stable'] and n['stable']]
 out={'n':len(ids)}
 for name,path in [(a.baseline_label,baseline_suite),(a.label,root/'force_suite')]:
  f=dict(np.load(path/'metrics/forced.npz'));z=dict(np.load(path/'metrics/zero.npz'));names=f['joint_names'].tolist();values=[];ipsi=[]
  for i in ids:
   side=old['records'][i]['group'];other='right' if side=='left' else 'left'
   delta=(f['joint_pos'][100:301,i]-f['joint_pos'][100,i])-(z['joint_pos'][100:301,i]-z['joint_pos'][100,i])
   for which,arr in [(other,values),(side,ipsi)]:
    js=[j for j,n in enumerate(names) if n.startswith(which+'_') and any(k in n for k in ('shoulder','elbow','wrist'))]
    arr.append(float(np.abs(delta[:,js]).mean()))
  out[name]={'unforced_arm_q_mae_rad':float(np.mean(values)),'forced_arm_q_mae_rad':float(np.mean(ipsi)),'ratio':float(np.mean(values)/np.mean(ipsi))}
 result['matched_survivor_hand_response'][force]=out
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
