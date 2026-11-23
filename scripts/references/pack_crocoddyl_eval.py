"""Pack validated double-support references for the standard compliance worker."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/compliance_dataset')]
from pack import allocate, derivatives, normalized_qpos, G1_JOINT_NAMES, SurfaceFK, quat_apply


def pack_reference(folder, fk, z_offset):
    summary = json.loads((folder / 'result.json').read_text())
    if (summary['ddp_converged'] != summary['ddp_total']
            or summary['feasible_dynamics_samples'] != summary['dynamics_samples']
            or summary['max_foot_position_error_m'] >= .001
            or summary['min_joint_limit_margin_rad'] <= 0):
        raise ValueError(f'Reference failed convergence, support, or joint-limit checks: {folder}')
    with np.load(folder / 'trajectory.npz') as source:
        if not np.allclose(np.diff(source['time_s']), .01):
            raise ValueError(f'Expected a 100 Hz source trajectory: {folder}')
        q_pin = source['q_pinocchio'][::2]
        order = [list(source['joint_names']).index(joint) for joint in G1_JOINT_NAMES]
    qpos = np.column_stack((q_pin[:, :3], q_pin[:, [6, 3, 4, 5]], q_pin[:, 7:][:, order]))
    # One second standing, the generated motion, and two seconds at the endpoint.
    qpos = np.concatenate((np.repeat(qpos[:1], 50, axis=0), qpos,
                           np.repeat(qpos[-1:], 100, axis=0)))
    qpos = torch.from_numpy(normalized_qpos(qpos))
    qpos[:, 2] -= z_offset
    pos, quat = fk(qpos[:, 7:], qpos[:, :3], qpos[:, 3:7])
    inv = qpos[:, 3:7].clone()
    inv[:, 1:] *= -1
    body_pos = quat_apply(inv[:, None].expand(-1, len(fk.bodies), -1),
                          pos - qpos[:, None, :3])
    linear, angular, joint = map(torch.from_numpy, derivatives(qpos.numpy()))
    n = len(qpos)
    motion_dir = folder / 'motion'
    dst = allocate(motion_dir, n, len(fk.bodies))
    fields = dict(motion_id=torch.zeros(n, dtype=torch.int32), step=torch.arange(n, dtype=torch.int32),
                  root_pos_w=qpos[:, :3], root_quat_w=qpos[:, 3:7], root_lin_vel_w=linear,
                  root_ang_vel_w=angular, joint_pos=qpos[:, 7:], joint_vel=joint,
                  body_pos_w=pos, body_pos_b=body_pos, body_quat_w=quat)
    for key, value in fields.items():
        getattr(dst, key).copy_(value)
    metadata = dict(body_names=fk.bodies, joint_names=G1_JOINT_NAMES, starts=[0], ends=[n], fps=50, info={})
    (motion_dir / 'meta_motion.json').write_text(json.dumps(metadata))
    row = dict(name=folder.name, split='diagnostic', dataset=str(motion_dir.resolve()),
               clip=0, source_motion_id=0, source_start_step=0, frames=n,
               category=folder.name.split('_')[0])
    (folder / 'manifest.json').write_text(json.dumps(dict(version='compliance-v1.5',
                                                          seeds=[20261104], clips=[row]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    parser.add_argument('--z-offset', type=float, default=.03540915,
                        help='Z offset the target evaluation loader adds back')
    args = parser.parse_args()
    geometry = json.loads((ROOT / 'artifacts/ceer_segmented_jacobian/geometry.json').read_text())
    fk = SurfaceFK(geometry, G1_JOINT_NAMES, 'cpu')
    for folder in args.folders:
        pack_reference(folder, fk, args.z_offset)


if __name__ == '__main__':
    main()
