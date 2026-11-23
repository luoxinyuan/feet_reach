"""Summarize sparse contact-budget tradeoffs from the online Jacobian audit."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_samples(directory, prefix):
    parts = []
    seeds = []
    folders = sorted(path for path in directory.glob(f'{prefix}_seed*') if path.is_dir())
    for folder in folders:
        meta = json.loads((folder / 'result.json').read_text())
        with np.load(folder / 'jacobian_contact_sweep.npz') as samples, np.load(folder / 'traces.npz') as traces:
            ids = samples['env_id']
            step = samples['step']
            keep = traces['valid'][step, ids] & (samples['target_norm'] >= .001)
            part = {key: samples[key][keep] for key in samples.files}
        part['category'] = np.asarray([meta['rows'][i]['category'] for i in part['env_id']])
        parts.append(part)
        seeds.append(dict(seed=meta['seed'], samples=len(part['step']), checkpoint=meta['checkpoint']))
    if not parts:
        raise ValueError(f'No sampled rollouts for {prefix}')
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}, seeds


def distribution(count, bins):
    return np.bincount(count, minlength=bins).tolist()


def sweep(samples, threshold):
    error = samples['relative_error']
    best = error.min(axis=-1)
    # Compare to the globally best of 1/2/3. Post-processing can break nesting.
    selected = (error <= best[:, None] + threshold + 1e-7).argmax(axis=-1)
    row = np.arange(len(selected))
    added_error = error[row, selected] - best
    return dict(threshold=threshold,
                selected_1_to_3=distribution(selected, 3),
                proposed_force_count_0_to_3=distribution(samples['actual_count'][row, selected], 4),
                average_points=float(np.mean(selected + 1)),
                mean_added_relative_error=float(added_error.mean()),
                p90_added_relative_error=float(np.quantile(added_error, .9)),
                average_force_sum_N=float(samples['force_sum_N'][row, selected].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    thresholds = (0., .005, .01, .015, .02, .025, .05, .075, .10, .125, .15, .20, .25, .30)
    report = dict(
        selection_rule='smallest k whose relative residual is within epsilon of the best over all k',
        relative_error='norm(W @ (A @ f - target)) / max(norm(W @ target), 0.001)',
        samples='native contact selection times with a non-negligible weighted target and a live rollout',
    )
    for label in ('3kp', '9kp'):
        samples, seeds = load_samples(args.directory, label)
        errors = samples['relative_error']
        if not np.isfinite(errors).all():
            raise ValueError(f'Non-finite errors for {label}')
        native = distribution(samples['native_count'] - 1, 3)
        results = [sweep(samples, threshold) for threshold in thresholds]
        report[label] = dict(
            n=len(errors), native_selected_1_to_3=native, seeds=seeds,
            mean_errors_for_1_2_3=errors.mean(axis=0).tolist(),
            median_errors_for_1_2_3=np.median(errors, axis=0).tolist(),
            mean_oracle_error=float(errors.min(axis=1).mean()),
            median_target_norm=float(np.median(samples['target_norm'])),
            sweep=results,
            by_category={category: [sweep({k: v[samples['category'] == category]
                                          for k, v in samples.items()}, threshold)
                                    for threshold in thresholds]
                         for category in np.unique(samples['category'])},
        )
        print(label, 'n', len(errors), 'native', native, 'mean k error', errors.mean(axis=0))
        for r in results:
            p = 100*np.asarray(r['selected_1_to_3'])/len(errors)
            print(f"{r['threshold']:5.3f}: {np.round(p, 1)} %; "
                  f"mean added {r['mean_added_relative_error']:.4f}; "
                  f"p90 added {r['p90_added_relative_error']:.4f}; "
                  f"points {r['average_points']:.3f}")
    (args.directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
