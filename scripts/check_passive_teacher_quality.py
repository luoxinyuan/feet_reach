"""Prevent distilling an inaccurate teacher; thresholds are experimental criteria."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evaluation_dir',type=Path);a=p.parse_args()
guided=json.loads((a.evaluation_dir/'guided/summary.json').read_text())
hold=json.loads((a.evaluation_dir/'hold/summary.json').read_text())
checks={
 'moving_completion_at_least_95pct':guided['completion_fraction']>=.95,
 'moving_world_hand_error_at_most_8cm':guided['sample_weighted_metrics']['ee_error_m']<=.08,
 'moving_joint_mae_at_most_015rad':guided['sample_weighted_metrics']['joint_mae_rad']<=.15,
 'hold_completion_at_least_95pct':hold['completion_fraction']>=.95,
 'hold_world_hand_error_at_most_8cm':hold['sample_weighted_metrics']['ee_error_m']<=.08,
}
report={'passed':all(checks.values()),'checks':checks,
 'coordinate_system':'Hand-error thresholds use world coordinates; root-relative errors are reported separately.',
 'metrics':{name:data['sample_weighted_metrics'] for name,data in [('moving',guided),('hold',hold)]}}
(a.evaluation_dir/'quality_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Teacher validation failed; student training has NOT been started.')
