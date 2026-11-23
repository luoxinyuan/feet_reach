#!/usr/bin/env python3
"""Render a clean MuJoCo concept figure for motion-conditioned force prompts."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = ROOT / 'deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml'
MOTION_DIR = ROOT / 'artifacts/compliance_eval/20261121-surface5-native-spring-forward-bend/motion'
TRACE = ROOT / ('artifacts/compliance_eval/20261118-supermimic-root3-jacobian-latent-increment-curriculum-'
                'compliancemix-gpu567-v1_5/metrics/crocoddyl_suite_bend_forward_45deg/traces.npz')
OUT = ROOT / 'artifacts/force_mapping_eval/force_prompt_motion_transition.png'
TRACE_ROOT = ROOT / ('artifacts/compliance_eval/20261118-supermimic-root3-jacobian-latent-increment-curriculum-'
                     'compliancemix-gpu567-v1_5/metrics')
TRACE_CASES = {
    'forward_bend': 'crocoddyl_suite_bend_forward_45deg',
    'backward_bend': 'crocoddyl_suite_bend_backward_35deg',
    'left_bend': 'crocoddyl_suite_bend_left_35deg',
    'right_bend': 'crocoddyl_suite_bend_right_35deg',
    'crouch': 'crocoddyl_suite_crouch_20cm',
    'yaw_left': 'crocoddyl_suite_yaw_left_30deg',
}
REFERENCE_CASES = {
    'forward_bend': 'artifacts/compliance_eval/20261121-surface5-native-spring-forward-bend/motion',
    'backward_bend': 'artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5/references_native/large_bend_sweep/bend_backward_35deg/motion',
    'left_bend': 'artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5/references_native/large_bend_sweep/bend_left_35deg/motion',
    'right_bend': 'artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5/references_native/large_bend_sweep/bend_right_35deg/motion',
    'crouch': 'artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5/references_native/crouch/motion',
    'yaw_left': 'artifacts/compliance_eval/20261118-supermimic-surface5-latent-increment-curriculum-legacy-gpu234-retry1-v1_5/references_native/difficulty_sweep/yaw_left_30deg/motion',
}
JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint',
    'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint',
    'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]
FORCE_COLORS = [(0.10, 0.55, 0.92, 1.0), (0.98, 0.42, 0.16, 1.0), (0.44, 0.40, 0.86, 1.0)]


def font(size: int, bold: bool = False):
    base = Path('/usr/share/fonts/truetype/dejavu')
    path = base / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def read_motion(frame: int):
    meta = json.loads((MOTION_DIR / '_tensordict/meta.json').read_text())

    def array(key):
        item = meta[key]
        dtype = {'torch.float32': np.float32, 'torch.int32': np.int32}[item['dtype']]
        return np.memmap(MOTION_DIR / '_tensordict' / f'{key}.memmap', mode='r',
                         dtype=dtype, shape=tuple(item['shape']))

    return array('root_pos_w')[frame].copy(), array('root_quat_w')[frame].copy(), array('joint_pos')[frame].copy()


def make_model(width: int, height: int):
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ROBOT_XML.parent / 'meshes'))
    for sky in root.findall('.//texture[@type="skybox"]'):
        sky.set('builtin', 'flat'); sky.set('rgb1', '0.86 0.88 0.91'); sky.set('rgb2', '0.86 0.88 0.91')
    for texture in root.findall('.//texture[@name="groundplane"]'):
        texture.set('builtin', 'flat'); texture.set('rgb1', '0.91 0.925 0.94'); texture.set('rgb2', '0.91 0.925 0.94')
        texture.attrib.pop('mark', None); texture.attrib.pop('markrgb', None)
    for light in root.findall('.//light'):
        light.set('diffuse', '0.8 0.82 0.86'); light.set('specular', '0.22 0.22 0.22')
    for world in root.findall('.//worldbody'):
        for floor in list(world.findall('./geom[@name="floor"]')):
            world.remove(floor)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.vis.global_.offwidth = width; model.vis.global_.offheight = height
    model.vis.quality.shadowsize = 2048
    model.opt.timestep = 0.02
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=5000)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.82]
    camera.distance = 2.15; camera.azimuth = 245.; camera.elevation = -8.
    return model, data, renderer, camera


def set_pose(model, data, root_pos, root_quat_wxyz, joints):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.
    free = model.joint('floating_base_joint').id
    adr = model.jnt_qposadr[free]
    data.qpos[adr:adr + 3] = root_pos
    data.qpos[adr + 3:adr + 7] = root_quat_wxyz
    for name, value in zip(JOINT_NAMES, joints):
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)


def add_force_markers(model, data, renderer):
    names = ['pelvis', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
    local = [np.zeros(3), np.array([.116, .000016, 0.]), np.array([.116, -.000016, 0.])]
    points = []
    for i, (name, offset) in enumerate(zip(names, local)):
        body = model.body(name).id
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
        points.append(point.copy())
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        color = np.asarray(FORCE_COLORS[i], dtype=np.float32)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), color)
        geom.size[:] = [.028, .028, .028]
        geom.pos[:] = point
        renderer.scene.ngeom += 1
    return points


def project_to_pixels(point, camera, width, height):
    pos = np.asarray(camera.pos, dtype=float)
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    relative = np.asarray(point, dtype=float) - pos
    depth = float(relative @ forward)
    half_height = float(camera.frustum_top) * depth / float(camera.frustum_near)
    aspect = width / height
    half_width = half_height * aspect
    x = width * (.5 + .5 * float(relative @ right) / half_width)
    y = height * (.5 - .5 * float(relative @ up) / half_height)
    return np.array([x, y]), right, up


def draw_projected_forces(image, points, camera, forces):
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    for i, (point, force) in enumerate(zip(points, forces)):
        origin, right, up = project_to_pixels(point, camera, width, height)
        # Project the world-frame force into the camera plane, then normalize
        # for legibility; actual force magnitudes are printed in the legend.
        direction = np.array([force @ right, -(force @ up)], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-5:
            continue
        direction /= norm
        magnitude = float(np.linalg.norm(force))
        length = 42. + 20. * magnitude
        end = origin + direction * length
        color = tuple(int(255 * c) for c in FORCE_COLORS[i][:3]) + (255,)
        draw.line((tuple(origin), tuple(end)), fill=color, width=8)
        perp = np.array([-direction[1], direction[0]])
        head_base = end - direction * 19.
        triangle = [tuple(end), tuple(head_base + perp * 11.), tuple(head_base - perp * 11.)]
        draw.polygon(triangle, fill=color)
        draw.ellipse((origin[0]-9, origin[1]-9, origin[0]+9, origin[1]+9),
                     fill=color, outline=(255, 255, 255, 255), width=3)


def render_pose(model, data, renderer, camera, pose, force=None, tint=None):
    set_pose(model, data, *pose)
    camera.lookat[:] = [pose[0][0], pose[0][1], .82]
    # MuJoCo mesh geoms are silver in the source asset. A restrained blue tint
    # distinguishes the next pose while keeping the robot recognizable.
    original = model.geom_rgba.copy()
    if tint is not None:
        for geom in range(model.ngeom):
            if model.geom_group[geom] == 1 and model.geom_bodyid[geom] != 0:
                model.geom_rgba[geom] = tint
    renderer.update_scene(data, camera)
    points = None
    if force is not None:
        points = add_force_markers(model, data, renderer)
    image = renderer.render().copy()
    camera_frame = renderer.scene.camera[0]
    model.geom_rgba[:] = original
    image = Image.fromarray(image)
    if force is not None:
        draw_projected_forces(image, points, camera_frame, force)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--case', choices=tuple(TRACE_CASES), default='forward_bend')
    parser.add_argument('--before-frame', type=int, default=0)
    parser.add_argument('--after-frame', type=int, default=240)
    parser.add_argument('--force-frame', type=int, default=240)
    parser.add_argument('--azimuth', type=float, default=220.)
    parser.add_argument('--width', type=int, default=1000)
    parser.add_argument('--height', type=int, default=1000)
    args = parser.parse_args()

    trace_path = TRACE_ROOT / TRACE_CASES[args.case] / 'traces.npz'
    with np.load(trace_path) as trace:
        force = trace['point_forces_world_N'][args.force_frame, 0].astype(np.float64)
    reference_dir = ROOT / REFERENCE_CASES[args.case] / '_tensordict'
    meta = json.loads((reference_dir / 'meta.json').read_text())
    def reference_array(key):
        item = meta[key]
        dtype = np.float32 if item['dtype'] == 'torch.float32' else np.int32
        return np.memmap(reference_dir / f'{key}.memmap', mode='r', dtype=dtype,
                         shape=tuple(item['shape']))
    roots = reference_array('root_pos_w')
    quats = reference_array('root_quat_w')
    joints = reference_array('joint_pos')
    after = (roots[args.after_frame].copy(), quats[args.after_frame].copy(),
             joints[args.after_frame].copy())
    # Align root translation while preserving each target body's orientation.
    root_before = roots[args.before_frame].copy()
    root_before[:2] = after[0][:2]  # align horizontal placement; retain crouch height
    before = (root_before, quats[args.before_frame].copy(),
              joints[args.before_frame].copy())
    model, data, renderer, camera = make_model(args.width, args.height)
    camera.azimuth = args.azimuth
    camera.lookat[:] = [after[0][0], after[0][1], .86]
    # Render the destination pose as the solid robot, then composite a
    # translucent lavender source-pose silhouette over the same location.
    solid = render_pose(model, data, renderer, camera, after,
                        tint=(.63, .70, .77, 1.0))
    ghost = render_pose(model, data, renderer, camera, before,
                        tint=(.49, .38, .76, 1.0))
    set_pose(model, data, *after)
    renderer.update_scene(data, camera)
    points = add_force_markers(model, data, renderer)
    camera_frame = renderer.scene.camera[0]
    renderer.close()
    # Find robot pixels against the renderer's flat background and blend the
    # source pose over the solid destination pose as a translucent ghost.
    background = np.asarray(solid.getpixel((0, 0)), dtype=np.int16)
    ghost_rgb = np.asarray(ghost, dtype=np.int16)
    ghost_mask = np.max(np.abs(ghost_rgb - background[None, None, :]), axis=2) > 7
    solid_rgb = np.asarray(solid, dtype=np.float32)
    blend = solid_rgb.copy()
    blend[ghost_mask] = .22 * ghost_rgb[ghost_mask] + .78 * solid_rgb[ghost_mask]
    canvas = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8), mode='RGB')
    # Add the actual target-frame contacts and projected online forces above
    # both poses. Force vectors are direction-faithful; visual lengths are
    # normalized so the small hand force remains visible.
    draw_projected_forces(canvas, points, camera_frame, force)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(args.output)
    print('case=', args.case, 'reference_frames=', [args.before_frame, args.after_frame],
          'forces_N=', json.dumps(force.tolist()))


if __name__ == '__main__':
    main()
