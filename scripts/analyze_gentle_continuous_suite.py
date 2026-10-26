"""Compare continuous-force experiments to paired historical raw traces."""
import argparse
import json
from pathlib import Path
import numpy as np

MODELS = ['CEER', 'A_teacher', 'A_student', 'B_teacher', 'B_student',
          'C_teacher', 'C_student', 'D_teacher', 'D_student']
METRICS = ['peak_force_N', 'max_sliding_impulse_100ms_Ns', 'impulse_100ms_Ns',
           'max_rate_N_s', 'precontact_vx_m_s', 'hold_force_N',
           'hold_force_std_N', 'hold_contact_fraction', 'longest_hold_contact_s',
           'required_contact_fraction', 'detachments_20ms', 'rise_variation_N',
           'hold_tangent_error_m', 'residual_mean_m']


def longest(mask):
    result = current = 0
    for value in mask:
        current = current + 1 if value else 0
        result = max(result, current)
    return result


def extra_metrics(physics, duration, wall, stable):
    f = physics[:, 1]
    hold = physics[:, 0] > duration
    required = physics[:, 5] > wall + .005
    contact = f > 1.
    # Complete windows anywhere in the trial, including later collisions.
    impulse = np.convolve(f, np.ones(20) * .005, mode='valid') if len(f) >= 20 else []
    consecutive_off = detachments = 0
    touched = False
    for force, req in zip(f, required):
        if not req:
            touched = False
            consecutive_off = 0
            continue
        if force > 1.:
            touched = True
            consecutive_off = 0
        elif force < .5 and touched:
            consecutive_off += 1
            if consecutive_off == 4:
                detachments += 1
        else:
            consecutive_off = 0
    good_hold = contact & hold & required
    return dict(max_sliding_impulse_100ms_Ns=float(np.max(impulse)) if len(impulse) else None,
                hold_contact_fraction=float(contact[hold].mean()) if hold.any() else None,
                hold_force_std_N=float(f[hold].std()) if hold.any() else None,
                longest_hold_contact_s=longest(good_hold) * .005,
                contact_hold_success=bool(stable and longest(good_hold) >= 60),
                detachments_20ms=detachments,
                rise_variation_N=float(np.maximum(np.diff(f, prepend=0), 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, default=Path('artifacts/gentle_matched_20261019'))
    parser.add_argument('--current', type=Path, default=Path('artifacts/gentle_matched_20261020'))
    args = parser.parse_args()
    report = {'models': {}, 'pairing': {}, 'comparisons': {}, 'records': {}}
    baseline = None
    for name in MODELS:
        folder = (args.current if name.startswith(('C_', 'D_')) else args.previous) / name
        rows = json.loads((folder / 'summary.json').read_text())
        assert len(rows) == 30, (name, len(rows))
        if baseline is None:
            baseline = rows
        assert [r['trial'] for r in rows] == [r['trial'] for r in baseline]
        differences = []
        exact = 0
        for row, base in zip(rows, baseline):
            data = np.load(folder / (row['trial'] + '.npz'))
            row.update(extra_metrics(data['physics'], row['duration'], row['wall'], row['stable']))
            initial = np.load(args.previous / 'CEER' / (base['trial'] + '.npz'))['initial']
            differences.append(float(np.abs(data['initial'] - initial).max()))
            exact += row['initial_sha256'] == base['initial_sha256']
        report['pairing'][name] = {'exact_matches': exact, 'max_initial_difference': max(differences)}
        stats = {}
        for key in METRICS:
            values = [r[key] for r in rows if r['stable'] and r['contact'] and r[key] is not None]
            stats[key] = dict(n=len(values), median=float(np.median(values)),
                              p90=float(np.percentile(values, 90)), mean=float(np.mean(values))) if values else None
        report['models'][name] = dict(n=len(rows), stable=sum(r['stable'] for r in rows),
            contact=sum(r['contact'] for r in rows),
            contact_hold_success=sum(r['contact_hold_success'] for r in rows),
            precision_success=sum(r['success'] for r in rows), stats=stats,
            hold_contact_90pct=sum(r['stable'] and r['hold_contact_fraction'] is not None and r['hold_contact_fraction'] >= .9 for r in rows),
            maximum_observed_peak_N=max(r['peak_force_N'] for r in rows),
            trials_peak_over_100N=sum(r['peak_force_N'] > 100 for r in rows),
            trials_peak_over_200N=sum(r['peak_force_N'] > 200 for r in rows),
            mean_reward_returns={k: float(np.mean([r['reward_returns'][k] for r in rows]))
                                 for k in rows[0]['reward_returns']})
        report['records'][name] = rows
    for a, b in [('CEER', 'C_student'), ('CEER', 'D_student'), ('A_student', 'C_student'),
                 ('B_student', 'C_student'), ('B_student', 'D_student'), ('C_student', 'D_student'),
                 ('C_teacher', 'C_student'), ('D_teacher', 'D_student')]:
        pairs = list(zip(report['records'][a], report['records'][b]))
        comparison = {}
        for key in METRICS:
            delta = np.array([y[key]-x[key] for x,y in pairs if x['stable'] and y['stable']
                              and x['contact'] and y['contact'] and x[key] is not None and y[key] is not None])
            comparison[key] = dict(n=len(delta), b_lower=int((delta < 0).sum()),
                median_delta=float(np.median(delta))) if len(delta) else None
        comparison['contact_success_lost'] = sum(x['contact_hold_success'] and not y['contact_hold_success'] for x,y in pairs)
        comparison['contact_success_gained'] = sum(not x['contact_hold_success'] and y['contact_hold_success'] for x,y in pairs)
        report['comparisons'][a+' -> '+b] = comparison
    (args.current/'continuous_analysis.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['CEER', 'A_student', 'B_student', 'C_student', 'D_student']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ['peak_force_N', 'max_sliding_impulse_100ms_Ns', 'hold_contact_fraction'],
            ['Whole-trial peak (N)', 'Maximum sliding 100 ms impulse (N s)', 'Hold contact fraction']):
        ax.boxplot([[r[key] for r in report['records'][n] if r['stable'] and r[key] is not None] for n in names],
                   labels=names, showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(args.current/'continuous_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'pairing': report['pairing'], 'models': report['models']}, indent=2))


if __name__ == '__main__':
    main()
