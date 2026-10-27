#!/usr/bin/env python3
"""Render a frontal MuJoCo video of the multi10 policy under hand forces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_whole_body_student import DEFAULT_MODEL, POLICY_BUNDLES, ROOT, make_simulation

PHASES = [
    ('No force', (0., 0., 0.), 1.0),
    ('Left hand: 5 N toward robot-left', (0., 5., 0.), 2.0),
    ('Left hand: 10 N toward robot-left', (0., 10., 0.), 2.0),
    ('Left hand: 10 N toward robot-right', (0., -10., 0.), 2.0),
    ('Left hand: 8 N upward', (0., 0., 8.), 2.0),
    ('Left hand: 8 N downward', (0., 0., -8.), 2.0),
    ('Left hand: 12 N forward', (12., 0., 0.), 2.0),
    ('No force', (0., 0., 0.), 1.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['segmented-multi10'], default='segmented-multi10')
    parser.add_argument('--bundle', type=Path, help='Override the selected exported MuJoCo policy bundle')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, required=True, help='Output MP4 path')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--azimuth', type=float, default=0., help='Camera azimuth; 0 is the robot front for the bundled G1 model')
    parser.add_argument('--distance', type=float, default=2.7)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    bundle = args.bundle or POLICY_BUNDLES[args.policy]
    sim = make_simulation(bundle, args.model, args.policy)
    sim.model.vis.global_.offwidth = args.width
    sim.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width, max_geom=5000)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0., 0., .85]
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = -8.
    point_i = sim.point_names.index('left_hand')
    dt = sim.model.opt.timestep
    phase_steps = [round(seconds/dt) for _, _, seconds in PHASES]
    phase_edges = np.cumsum([0] + phase_steps)
    total_steps = int(phase_edges[-1])
    frames_per_step = max(1, round(1/(args.fps*dt)))
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    writer = imageio.get_writer(args.output, fps=args.fps, codec='libx264', quality=8, macro_block_size=1)
    max_force = 0.; min_height = float('inf'); rows = []; video_frames = 0
    try:
        for step in range(total_steps):
            phase = min(int(np.searchsorted(phase_edges[1:], step, side='right')), len(PHASES)-1)
            desired = np.zeros_like(sim.forces)
            desired[point_i] = PHASES[phase][1]
            sim.step(desired)
            force = sim.forces[point_i].copy()
            magnitude = float(np.linalg.norm(force)); max_force = max(max_force, magnitude)
            min_height = min(min_height, float(sim.data.qpos[2]))
            if step % frames_per_step == 0 or step == total_steps-1:
                camera.lookat[:] = sim.data.qpos[:3] + np.array([0., 0., .10])
                renderer.update_scene(sim.data, camera)
                start = sim.point_positions()[point_i]
                if magnitude > .05:
                    end = start + force/magnitude*min(.85, .08*magnitude)
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array([1., .68, .08, 1.], dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .018, start, end)
                    renderer.scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image, 'RGBA')
                draw.rectangle((0, 0, args.width, 94), fill=(10, 15, 24, 225))
                draw.text((18, 8), 'ONE ROBOT  |  FRONTAL MuJoCo VIEW  |  MULTI10 SEGMENTED POLICY', font=small, fill=(190, 225, 255))
                draw.text((18, 37), f'{PHASES[phase][0]}  |  applied {magnitude:.1f} N  |  {sim.data.time:.1f} / {total_steps*dt:.1f} s', font=font, fill='white')
                draw.rectangle((14, 108, 420, 164), fill=(10, 15, 24, 210))
                draw.text((27, 117), 'Yellow arrow starts at the left hand', font=small, fill=(255, 210, 80))
                draw.text((27, 140), 'Arrow shows actual force direction (8 cm/N)', font=small, fill='white')
                writer.append_data(np.asarray(image)); video_frames += 1
            if step % frames_per_step == 0:
                rows.append({'time_s': float(sim.data.time), 'phase': PHASES[phase][0],
                    'left_hand_force_world_N': force.tolist(),
                    'left_hand_position_world_m': sim.point_positions()[point_i].tolist(),
                    'root_height_m': float(sim.data.qpos[2])})
    finally:
        writer.close(); renderer.close()
    summary = {'policy': args.policy, 'bundle': str(bundle), 'force_point': 'left_hand',
        'duration_s': total_steps*dt, 'fps': args.fps, 'frames': video_frames,
        'maximum_applied_force_N': max_force, 'minimum_root_height_m': min_height,
        'first_fall_time_s': sim.first_fall, 'phases': [
            {'label': label, 'target_force_world_N': list(force), 'duration_s': sec}
            for label, force, sec in PHASES]}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    args.output.with_name(args.output.stem+'_trace.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
