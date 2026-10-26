"""A short-clip teacher screen, not a claim of general compliance success."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
d=json.loads((args.directory/'guided/summary.json').read_text());m=d['sample_weighted_metrics']
checks={'completion_at_least_95pct':d['completion_fraction']>=.95,
        'joint_mae_at_most_015rad':m['joint_mae_rad']<=.15,
        'world_keypoint_error_at_most_015m':m['keypoint_error_m']<=.15}
report={'passed':all(checks.values()),'checks':checks,'metrics':m}
(args.directory/'quality_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Teacher gate failed; no student training started.')
