#!/usr/bin/env python3
"""Headless MuJoCo student viewer for VS Code port forwarding (stdlib HTTP)."""
from __future__ import annotations
import os
import sys
# Choose a platform-supported backend before MuJoCo is imported.
# Linux servers use EGL; desktops use their native OpenGL implementation.
if sys.platform == 'darwin':
    default_gl = 'cgl'
elif sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    default_gl = 'egl'
else:
    default_gl = 'glfw'
os.environ.setdefault('MUJOCO_GL', default_gl)
os.environ.setdefault('LP_NUM_THREADS', '2')
import argparse
import io
import json
import math
import secrets
import signal
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image
from mujoco_whole_body_student import (
    DEFAULT_BUNDLE, DEFAULT_MODEL, ROOT, POINT_NAMES, Interactive, Simulation, make_simulation, mujoco, np,
)


class WebInteraction(Interactive):
    """Reuse the desktop force law/drag plane without creating a GLFW window."""
    def __init__(self, sim, width, height):
        self.sim = sim
        self.width, self.height = width, height
        sim.model.vis.global_.offwidth = width
        sim.model.vis.global_.offheight = height
        self.renderer = mujoco.Renderer(sim.model, width=width, height=height, max_geom=5000)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0, 0, .8]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.2, 135, -18
        self.opt = mujoco.MjvOption()
        self.scene = self.renderer.scene
        self.paused = True
        self.follow = True
        self.selected = None
        self.dragging = False
        self.goal = None
        self.last_mouse = (0., 0.)
        self.refresh_scene()

    def refresh_scene(self):
        self.renderer.update_scene(self.sim.data, self.cam, self.opt)

    def ray(self, x, y):
        camera = self.scene.camera[0]
        eye = (self.scene.camera[0].pos + self.scene.camera[1].pos) / 2
        forward = np.array(camera.forward, dtype=float)
        forward /= np.linalg.norm(forward)
        up = np.array(camera.up, dtype=float)
        up /= np.linalg.norm(up)
        right = np.cross(forward, up)
        tangent = math.tan(math.radians(self.sim.model.vis.global_.fovy) / 2)
        direction = forward + (2*x/self.width-1)*(self.width/self.height)*tangent*right + (1-2*y/self.height)*tangent*up
        return eye, direction

    def handle(self, event):
        kind = event['type']
        if kind == 'release':
            self.dragging = False
        elif kind == 'reset':
            self.sim.reset()
            self.selected = None
            self.dragging = False
            self.refresh_scene()
        elif kind == 'pause':
            self.paused = bool(event['paused'])
            if self.paused:
                self.cancel_force()
        elif kind == 'follow':
            self.follow = not self.follow
        elif kind in ('down', 'move'):
            x, y = event['x']*self.width, event['y']*self.height
            if kind == 'down':
                self.dragging = False
                self.refresh_scene()
                hit = np.zeros(3)
                geom = np.array([-1], dtype=np.int32)
                flex, skin = geom.copy(), geom.copy()
                body = mujoco.mjv_select(self.sim.model, self.sim.data, self.opt,
                    self.width/self.height, x/self.width, 1-y/self.height, self.scene, hit, geom, flex, skin)
                if body > 0:
                    point = self.sim.nearest_point(body, hit)
                    if point is not None:
                        self.selected, self.dragging = point, True
                        self.plane_point = hit.copy()
                        self.plane_normal = np.array(self.scene.camera[0].forward, dtype=float)
                        self.anchor = self.sim.point_positions()[point].copy()
                        self.mouse_anchor = self.plane_hit(x, y)
                        self.goal = self.anchor.copy()
            elif self.dragging:
                self.goal = self.anchor + self.plane_hit(x, y) - self.mouse_anchor
            self.last_mouse = (x, y)
        elif kind == 'camera' and not self.dragging:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V if event.get('pan', False) else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            if event.get('pan', False):
                self.follow = False
            mujoco.mjv_moveCamera(self.sim.model, action, event['dx'], event['dy'], self.scene, self.cam)
        elif kind == 'zoom' and not self.dragging:
            mujoco.mjv_moveCamera(self.sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0, event['dy'], self.scene, self.cam)

    def cancel_force(self):
        self.dragging = False
        # While paused there are no physical substeps in which to decay the force.
        self.sim.forces[:] = 0
        self.sim.data.xfrc_applied[:] = 0

    def render_jpeg(self, quality):
        s = self.sim
        if self.follow and not self.dragging:
            self.cam.lookat[:] = s.data.qpos[:3]
        self.refresh_scene()
        for i, force in enumerate(s.forces):
            if np.linalg.norm(force) < .05:
                continue
            start = s.point_positions()[i]
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                np.eye(3).ravel(), np.array([1, .35, .05, 1], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .012, start, start+.02*force)
            self.scene.ngeom += 1
        if self.selected is not None:
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.025]*3),
                s.point_positions()[self.selected], np.eye(3).ravel(), np.array([.1, 1, .9, 1], dtype=np.float32))
            self.scene.ngeom += 1
            if hasattr(s, 'estimated_forces'):
                rot = s.data.xmat[s.root].reshape(3, 3)
                estimate = rot @ s.estimated_forces[self.selected] * 30.
                if np.linalg.norm(estimate) > .05:
                    start = s.point_positions()[self.selected] + np.array([0., 0., .04])
                    geom = self.scene.geoms[self.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array([.05, .65, 1., 1.], dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .01, start, start+.02*estimate)
                    self.scene.ngeom += 1
        frame = self.renderer.render()
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format='JPEG', quality=quality)
        return output.getvalue()


class Transport:
    def __init__(self, host, port):
        self.lock = threading.Lock()
        self.commands = deque(maxlen=128)
        self.jpeg = b''
        self.state = {}
        self.frame_id = 0
        self.owner = None
        self.heartbeat = 0.
        self.token = secrets.token_urlsafe(32)
        self.html = Path(__file__).with_name('mujoco_whole_body_web.html').read_text().replace('__TOKEN__', self.token).encode()
        transport = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body=b'', content_type='application/json', extra=None):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(len(body)))
                for key, value in (extra or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                with transport.lock:
                    if path == '/':
                        body, mime = transport.html, 'text/html; charset=utf-8'
                    elif path == '/frame.jpg':
                        body, mime = transport.jpeg, 'image/jpeg'
                    elif path == '/state':
                        body, mime = json.dumps(transport.state).encode(), 'application/json'
                    else:
                        self.reply(404); return
                    frame_id = transport.frame_id
                self.reply(200, body, mime, {'X-Frame-Id': frame_id})

            def do_POST(self):
                if self.path != '/command':
                    self.reply(404); return
                if self.headers.get('X-Control-Token') != transport.token:
                    self.reply(403); return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 2048:
                        raise ValueError('invalid payload length')
                    event = json.loads(self.rfile.read(length))
                    if not isinstance(event, dict):
                        raise ValueError('expected an object')
                    client = event.pop('client')
                    if not isinstance(client, str) or not 1 <= len(client) <= 80:
                        raise ValueError('invalid client id')
                    validate_event(event)
                except (ValueError, KeyError, TypeError):
                    self.reply(400); return
                now = time.monotonic()
                with transport.lock:
                    if transport.owner != client:
                        if transport.owner is not None and now-transport.heartbeat < 3:
                            self.reply(409, b'{"error":"Another browser tab has control"}'); return
                        transport.commands.clear()
                        transport.commands.append({'type': 'pause', 'paused': True})
                        transport.owner = client
                    transport.heartbeat = now
                    if event['type'] != 'heartbeat':
                        if len(transport.commands) >= 127:
                            self.reply(429); return
                        transport.commands.append(event)
                self.reply(204)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def take_events(self):
        with self.lock:
            connected = self.owner is not None and time.monotonic()-self.heartbeat < 3
            events = list(self.commands)
            self.commands.clear()
        return connected, events

    def publish(self, jpeg, state):
        with self.lock:
            self.jpeg, self.state = jpeg, state
            self.frame_id += 1

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def validate_event(event):
    kind = event.get('type')
    if kind not in {'heartbeat', 'down', 'move', 'release', 'reset', 'pause', 'follow', 'camera', 'zoom'}:
        raise ValueError('unknown command')
    fields = {'down': ['x', 'y'], 'move': ['x', 'y'], 'camera': ['dx', 'dy'], 'zoom': ['dy']}.get(kind, [])
    for key in fields:
        value = event[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10:
            raise ValueError('invalid pointer coordinate')
    if kind == 'pause' and not isinstance(event.get('paused'), bool):
        raise ValueError('invalid pause value')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--bundle', type=Path, help='Override the selected policy bundle')
    p.add_argument('--policy', choices=['direct','sonic','ceer','jacobian','segmented','segmented-static-hold','segmented-multi10','ceer-ee-root3','ceer-dense23'], default='direct')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=600)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--quality', type=int, default=80)
    args = p.parse_args()
    if not (1 <= args.fps <= 60 and 320 <= args.width <= 1920 and 200 <= args.height <= 1080 and 20 <= args.quality <= 95):
        p.error('Invalid FPS, image size or JPEG quality')
    # Bind before loading physics, so an occupied port fails clearly.
    transport = Transport(args.host, args.port)
    sim = make_simulation(args.bundle, args.model, args.policy)
    ui = WebInteraction(sim, args.width, args.height)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    transport.start()
    print(f'MuJoCo student: http://{args.host}:{args.port} | GL={os.environ["MUJOCO_GL"]} | paused at standing', flush=True)
    previous = time.monotonic()
    next_frame = previous
    accumulator = 0.
    wall_start, sim_start = previous, sim.data.time
    realtime = 0.
    try:
        while running:
            now = time.monotonic()
            accumulator += min(now-previous, .1)
            previous = now
            connected, events = transport.take_events()
            for event in events:
                ui.handle(event)
                if event['type'] == 'reset':
                    wall_start, sim_start = now, 0.
            if not connected:
                ui.paused = True
                ui.cancel_force()
            if ui.paused:
                accumulator = 0.
            # Yield to incoming mouse commands at least every 10 physics steps.
            for _ in range(min(10, int(accumulator/sim.model.opt.timestep))):
                sim.step(ui.desired_force())
                accumulator -= sim.model.opt.timestep
            if now >= next_frame:
                if now-wall_start >= 1:
                    realtime = (sim.data.time-sim_start)/(now-wall_start)
                    wall_start, sim_start = now, sim.data.time
                jpeg = ui.render_jpeg(args.quality)
                state = {'policy': sim.meta.get('policy_alias',sim.meta.get('backend','direct')), 'initial_height': sim.meta.get('root_height',.74), 'time': float(sim.data.time), 'paused': ui.paused, 'connected': connected,
                    'selected': sim.point_names[ui.selected] if ui.selected is not None else None,
                    'selected_estimated_force_N': (float(np.linalg.norm(sim.estimated_forces[ui.selected])*30)
                        if ui.selected is not None and hasattr(sim,'estimated_forces') else None),
                    'force_N': float(np.linalg.norm(sim.forces, axis=1).sum()),
                    'force_limit_N': 30, 'root_height': float(sim.data.qpos[2]),
                    'force_estimate_N': (float(np.linalg.norm(sim.estimated_forces,axis=1).sum()*30)
                        if hasattr(sim,'estimated_forces') else None),
                    'fall': sim.first_fall is not None, 'follow': ui.follow, 'dragging': ui.dragging,
                    'realtime_factor': realtime, 'width': args.width, 'height': args.height}
                transport.publish(jpeg, state)
                next_frame = max(next_frame+1/args.fps, time.monotonic())
            time.sleep(.001)
    finally:
        ui.cancel_force()
        transport.close()
        ui.renderer.close()


if __name__ == '__main__':
    main()
