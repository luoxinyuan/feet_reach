"""Measure bend, crouch, and yaw range with fixed Crocoddyl references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import imageio.v2 as iio
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def suite_cases(surface5_report, forward_manifest):
    base = surface5_report / 'references_native'
    yield 'bend_forward_45deg', 'forward', forward_manifest, 45
    for direction in ('backward', 'left', 'right'):
        case = f'bend_{direction}_45deg'
        yield case, direction, base / 'large_bend_sweep' / case / 'manifest.json', 45
    sweep = base / 'difficulty_sweep'
    yield 'crouch_23cm', 'crouch', sweep / 'crouch_23cm' / 'manifest.json', 23
    for direction in ('left', 'right'):
        motion = f'yaw_{direction}'
        case = f'{motion}_60deg'
        yield case, motion, sweep / case / 'manifest.json', 60


def metrics(path, motion):
    result = json.loads((path / 'result.json').read_text())
    seed = result.get('latent_eval_seed')
    alpha = result.get('latent_curriculum_alpha_eval')
    if alpha is not None and (alpha != 1.0 or not (
            str(seed).startswith('default standing-pose latent') or
            str(seed).startswith('none; policy outputs the full decoder latent'))):
        raise ValueError(f'Unexpected latent initialization: {path}')
    with np.load(path / 'traces.npz') as trace:
        root = trace['root'][:, 0]
        if motion == 'crouch':
            progress = 100 * (root[0, 2] - root[:, 2])
        else:
            if motion.startswith('yaw_'):
                angle = np.rad2deg(np.unwrap(trace['yaw'][:, 0]))
                sign = -1 if motion == 'yaw_right' else 1
            else:
                quat = trace['torso_quat_world'][:, 0, :][:, [1, 2, 3, 0]]
                axis = 0 if motion in ('left', 'right') else 1
                angle = Rotation.from_quat(quat).as_euler('xyz', degrees=True)[:, axis]
                sign = -1 if motion in ('left', 'backward') else 1
            progress = sign * (angle - angle[0])
        fall = int(trace['first_fall_step'][0])
        stop = len(progress) if fall < 0 else max(1, fall)
        peak = max(0.0, float(progress[:stop].max()))
        ankle = trace['ankle_pos_world'][:stop, 0, :, :2]
        if 'point_forces_world_N' in trace:
            point_forces = trace['point_forces_world_N'][:stop, 0]
            peak_force_sum = float(np.linalg.norm(point_forces, axis=-1).sum(axis=-1).max())
        elif 'force_sum_N' in trace:
            peak_force_sum = float(trace['force_sum_N'][:stop, 0].max())
        else:
            peak_force_sum = None
        return dict(
            peak=peak,
            max_completed=peak if fall < 0 else None,
            final=float(progress[-1]) if fall < 0 else None,
            pelvis_drift_cm=float(100 * np.linalg.norm(root[:stop, :2] - root[0, :2], axis=-1).max()),
            ankle_drift_cm=float(100 * np.linalg.norm(ankle - ankle[0], axis=-1).max()),
            peak_force_sum_N=peak_force_sum,
            fall_s=fall * result['dt'] if fall >= 0 else None,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-eval', required=True, type=Path)
    parser.add_argument('--forward-manifest', required=True, type=Path)
    parser.add_argument('--gpus', required=True, help='Comma-separated GPU IDs, one lane per entry')
    parser.add_argument('--videos-dir', default='new video')
    args = parser.parse_args()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve(strict=True)
    with checkpoint.open('rb') as file:
        checkpoint_hash = hashlib.file_digest(file, 'sha256').hexdigest() if hasattr(
            hashlib, 'file_digest') else hashlib.sha256(file.read()).hexdigest()
    cases = list(suite_cases(args.reference_eval, args.forward_manifest))
    if len(cases) != 7 or len({c[0] for c in cases}) != 7:
        raise ValueError('Expected seven distinct Crocoddyl motions')
    for _, _, manifest, _ in cases:
        manifest.resolve(strict=True)
    config_candidates = [checkpoint.parents[3] / 'task_config.yaml',
                         checkpoint.parents[3] / '.hydra' / 'config.yaml']
    config_text = '\n'.join(p.read_text() for p in config_candidates if p.is_file())
    latent_curriculum = 'LatentIncrementCurriculumAction' in config_text or 'DirectLatentCurriculumAction' in config_text
    for directory in ('logs', 'metrics', args.videos_dir):
        (output / directory).mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(',')

    def run_lane(gpu, assigned):
        errors = []
        for name, _, manifest, _ in assigned:
            result_path = output / 'metrics' / f'crocoddyl_suite_{name}'
            video_path = output / args.videos_dir / f'{name}_native_prompt.mp4'
            saved = result_path / 'result.json'
            if saved.is_file() and video_path.is_file():
                previous = json.loads(saved.read_text())
                if (previous['checkpoint_sha256'] != checkpoint_hash or previous['manifest_sha256']
                        != hashlib.sha256(manifest.read_bytes()).hexdigest()):
                    raise ValueError(f'Checkpoint or manifest changed for {name}')
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', MEMPATH=str(ROOT / 'dataset'))
            isaac = str(Path(os.environ.get('ISAACLAB_ROOT', str(ROOT.parent / 'IsaacLab'))) / 'source/isaaclab')
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (isaac, str(ROOT), env.get('PYTHONPATH'))))
            command = [sys.executable, str(ROOT / 'scripts/compliance_eval/worker.py'),
                       '--checkpoint', str(checkpoint), '--manifest', str(manifest.resolve()),
                       '--output', str(result_path), '--suite', 'tracking']
            if latent_curriculum:
                command.append('--tracking-standing-seed')
            command.extend(['--video-clip', '0', '--video-output', str(video_path)])
            print(f'START {name} GPU {gpu}', flush=True)
            with (output / 'logs' / f'crocoddyl_suite_{name}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'DONE {name} {result.returncode}', flush=True)
            if result.returncode:
                errors.append(name)
        return errors

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        lanes = [pool.submit(run_lane, gpu, cases[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        failures = [name for lane in lanes for name in lane.result()]
    if failures:
        raise RuntimeError(f'Failed Crocoddyl cases: {failures}')
    results = []
    for name, motion, _, target in cases:
        video = f'{args.videos_dir}/{name}_native_prompt.mp4'
        with iio.get_reader(output / video) as reader:
            if reader.count_frames() != 151 or reader.get_data(100).shape != (540, 960, 3):
                raise ValueError(f'Invalid video {video}')
        stats = metrics(output / 'metrics' / f'crocoddyl_suite_{name}', motion)
        results.append(dict(name=name, motion=motion, target=target, video=video, **stats))
    (output / 'crocoddyl_suite_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    section = [
        '## Crocoddyl motion suite with native prompts', '',
        'Each of the seven cases is a 6 s rollout. Latent-curriculum policies use their final autonomous '
        'stage (incremental policies start from a fixed standing latent; direct-latent policies use their '
        'full policy output). Other policies start from the trajectory initial state. All use the checkpoint’s '
        'training-time force rule. The reference starts with a 1 s hold, transitions over 3 s, and holds '
        'for 2 s. Maximum completed amplitude is the peak in the commanded direction for a rollout '
        'that survives to the end. A fallen trial shows its pre-fall peak separately. Negative final '
        'values indicate motion opposite to the target.', '',
        '| Motion and video | Reference target | Maximum completed | Pre-fall peak | Final | Pelvis / ankle XY drift | Fell |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for r in results:
        unit = ' cm' if r['motion'] == 'crouch' else '°'
        maximum = f"{r['max_completed']:.1f}{unit}" if r['max_completed'] is not None else '—'
        pre_fall = f"{r['peak']:.1f}{unit}" if r['fall_s'] is not None else '—'
        final = f"{r['final']:.1f}{unit}" if r['final'] is not None else '—'
        fall = f"{r['fall_s']:.2f} s" if r['fall_s'] is not None else 'No'
        section.append(f"| [{r['name']}](<{r['video']}>) | {r['target']:g}{unit} | {maximum} | "
                       f"{pre_fall} | {final} | {r['pelvis_drift_cm']:.1f} / {r['ankle_drift_cm']:.1f} cm | {fall} |")
    report = output / 'REPORT.md'
    old = report.read_text() if report.is_file() else '# Compliance evaluation\n'
    old = old.split('\n' + section[0], 1)[0].rstrip()
    report.write_text(old + '\n\n' + '\n'.join(section) + '\n')


if __name__ == '__main__':
    main()
