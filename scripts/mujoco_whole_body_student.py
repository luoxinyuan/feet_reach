#!/usr/bin/env python3
"""Target-free whole-body student in MuJoCo; left-drag upper body to apply force."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / 'deploy/mujoco_whole_body/policy'
DEFAULT_MODEL = ROOT / 'deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml'
POINT_NAMES = ['torso_link'] + [f'{side}_{part}' for side in ['left','right']
    for part in ['shoulder_roll_link','elbow_link','wrist_roll_link','hand_mimic']]


def capped(v, limit):
    return v * min(1., limit / max(float(np.linalg.norm(v)), 1e-12))


class Simulation:
    def __init__(self, bundle, model_path, physics_dt=None):
        torch.set_num_threads(1)
        self.meta = json.loads((bundle/'metadata.json').read_text())
        self.policy = torch.jit.load(str(bundle/'student.pt'), map_location='cpu').eval()
        norm = torch.load(bundle/'normalization.pt', map_location='cpu', weights_only=True)
        self.loc, self.scale = norm['loc'], norm['scale']
        xml = ET.parse(model_path)
        xml.find('compiler').set('meshdir',str((model_path.parent/'meshes').resolve()))
        # These fixed, massless points correspond to the training force channels.
        # Rubber hands are fused into wrist_yaw in this MuJoCo asset.
        for side in ['left','right']:
            parent = xml.find(f".//body[@name='{side}_wrist_yaw_link']")
            if xml.find(f".//body[@name='{side}_hand_mimic']") is None:
                # FixedJoint localPos0 in g1_flat_fullcol.usd (not the rubber mesh origin).
                ET.SubElement(parent,'body',name=f'{side}_hand_mimic',pos=f'0.116 {0.000016093254 if side=="left" else -0.000016093254} 0.0000046119094')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(),encoding='unicode'))
        physics_dt = physics_dt or self.meta.get('physics_dt', .002)
        self.model.opt.timestep = physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 50
        self.data = mujoco.MjData(self.model)
        self.ids = np.array([self.model.joint(n).id for n in self.meta['joint_names']])
        self.qadr = self.model.jnt_qposadr[self.ids]
        self.vadr = self.model.jnt_dofadr[self.ids]
        self.act = np.array([self.model.actuator(n).id for n in self.meta['joint_names']])
        self.q0 = np.array(self.meta['default_pos'])
        self.kp, self.kd = np.array(self.meta['kp']), np.array(self.meta['kd'])
        self.effort = np.array(self.meta['effort_limit'])
        self.model.dof_armature[self.vadr] = self.meta['armature']
        self.model.dof_damping[self.vadr] = 0.
        self.model.dof_frictionloss[self.vadr] = 0.
        self.points = np.array([self.model.body(n).id for n in POINT_NAMES])
        self.root = self.model.body('pelvis').id
        self.decimation = round(self.meta['policy_dt']/physics_dt)
        assert abs(self.decimation*physics_dt-.02)<1e-9
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:3] = [0,0,self.meta.get('root_height',.74)]
        self.data.qpos[3:7] = [1,0,0,0]
        self.data.qpos[self.qadr] = self.q0
        mujoco.mj_forward(self.model,self.data)
        self.history = np.tile(self.q0,(5,1))  # absolute q, most recent first
        self.actions = np.zeros((3,29))
        self.target = self.q0.copy()
        self.forces = np.zeros((9,3))
        self.substep_q = [self.q0.copy(),self.q0.copy()]
        self.steps = 0
        self.last_obs = np.zeros(265)
        self.last_action = np.zeros(29)
        self.first_fall = None

    def observation(self):
        rot = self.data.xmat[self.root].reshape(3,3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.root,velocity,1)
        return np.concatenate([velocity[:3], rot.T @ np.array([0.,0.,-1.]),
            self.history.ravel(),self.actions.ravel(),(self.forces @ rot).ravel()]).astype(np.float32)

    def step(self, desired):
        # Observation and physics use the same limited force; no hidden wrench.
        for i in range(9):
            goal = capped(desired[i],self.meta['force_limit'])
            self.forces[i] += capped(goal-self.forces[i],self.meta['force_slew']*self.model.opt.timestep)
        self.data.xfrc_applied[:] = 0.
        for bid, force in zip(self.points,self.forces):
            # xfrc_applied acts at COM. Shift to body origin to match Isaac's application site.
            self.data.xfrc_applied[bid,:3] = force
            self.data.xfrc_applied[bid,3:] = np.cross(self.data.xpos[bid]-self.data.xipos[bid],force)
        if self.steps % self.decimation == 0:
            self._update_policy()
        self._before_torque()
        torque = self.kp*(self.target-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.vadr]
        self.data.ctrl[self.act] = np.clip(torque,-self.effort,self.effort)
        mujoco.mj_step(self.model,self.data)
        # mj_step integrates qpos/qvel after computing derived Cartesian fields.
        # Refresh them so picking, force frames and the next observation agree.
        mujoco.mj_forward(self.model,self.data)
        self.substep_q[self.steps%2] = self.data.qpos[self.qadr].copy()
        self.steps += 1
        if self.steps%self.decimation == 0:
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = np.mean(self.substep_q,axis=0)
        if self.first_fall is None and self.data.qpos[2] < .4:
            self.first_fall = float(self.data.time)

    def _update_policy(self):
        self.last_obs = self.observation()
        with torch.inference_mode():
            obs = (torch.from_numpy(self.last_obs)-self.loc)/self.scale
            action = self.policy(obs.unsqueeze(0))[0].numpy().clip(-10,10)
        if not np.isfinite(action).all():
            raise RuntimeError('Non-finite policy action')
        self.last_action = action.copy()
        self.actions[1:] = self.actions[:-1].copy()
        self.actions[0] = action
        self.target = self.q0 + self.meta['action_scale']*action

    def _before_torque(self):
        pass

    def nearest_point(self, body, hit):
        name = self.model.body(body).name
        if not any(x in name for x in ('torso','waist_pitch','shoulder','elbow','wrist','hand')):
            return None
        candidates = list(range(9))
        if name.startswith('left_'): candidates = list(range(1,5))
        elif name.startswith('right_'): candidates = list(range(5,9))
        else: candidates = [0]
        return min(candidates,key=lambda i:np.linalg.norm(self.data.xpos[self.points[i]]-hit))


class Interactive:
    def __init__(self,sim):
        import glfw
        self.glfw = glfw
        if not glfw.init(): raise RuntimeError('GLFW cannot open a display. Run on a graphical desktop or use --headless.')
        self.window = glfw.create_window(1280,800,'Whole-body student | mouse force',None,None)
        if not self.window: raise RuntimeError('Cannot create GLFW window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.sim = sim
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0,0,.8]
        self.cam.distance=3.2; self.cam.azimuth=135; self.cam.elevation=-18
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(sim.model,maxgeom=5000)
        self.context = mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
        self.paused=False; self.follow=True; self.selected=None; self.dragging=False
        self.goal=None; self.last_mouse=(0,0); self.frame=0
        self.update_scene()
        glfw.set_mouse_button_callback(self.window,self.mouse_button)
        glfw.set_cursor_pos_callback(self.window,self.mouse_move)
        glfw.set_scroll_callback(self.window,self.scroll)
        glfw.set_key_callback(self.window,self.key)

    def update_scene(self):
        mujoco.mjv_updateScene(self.sim.model,self.sim.data,self.opt,None,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.scene)

    def ray(self,x,y):
        w,h=self.glfw.get_window_size(self.window)
        camera=self.scene.camera[0]
        eye=(self.scene.camera[0].pos+self.scene.camera[1].pos)/2
        forward=np.array(camera.forward,dtype=float); forward/=np.linalg.norm(forward)
        up=np.array(camera.up,dtype=float); up/=np.linalg.norm(up)
        right=np.cross(forward,up)
        tangent=math.tan(math.radians(self.sim.model.vis.global_.fovy)/2)
        direction=forward+(2*x/w-1)*(w/h)*tangent*right+(1-2*y/h)*tangent*up
        return eye,direction

    def plane_hit(self,x,y):
        origin,direction=self.ray(x,y)
        denominator=np.dot(direction,self.plane_normal)
        if abs(denominator)<1e-6: return self.plane_point.copy()
        return origin+direction*np.dot(self.plane_point-origin,self.plane_normal)/denominator

    def mouse_button(self,window,button,action,mods):
        g=self.glfw
        x,y=g.get_cursor_pos(window); self.last_mouse=(x,y)
        if button != g.MOUSE_BUTTON_LEFT: return
        if action == g.RELEASE:
            self.dragging=False
            return
        w,h=g.get_window_size(window)
        hit=np.zeros(3); geom=np.array([-1],dtype=np.int32); flex=geom.copy(); skin=geom.copy()
        body=mujoco.mjv_select(self.sim.model,self.sim.data,self.opt,w/h,x/w,1-y/h,self.scene,hit,geom,flex,skin)
        if body<=0: return
        point=self.sim.nearest_point(body,hit)
        if point is None: return
        self.selected=point; self.dragging=True
        self.plane_point=hit.copy(); self.plane_normal=np.array(self.scene.camera[0].forward,dtype=float)
        self.anchor=self.sim.data.xpos[self.sim.points[point]].copy()
        self.mouse_anchor=self.plane_hit(x,y)
        self.goal=self.anchor.copy()

    def mouse_move(self,window,x,y):
        g=self.glfw
        dx=x-self.last_mouse[0]; dy=y-self.last_mouse[1]; self.last_mouse=(x,y)
        if self.dragging:
            self.goal=self.anchor+self.plane_hit(x,y)-self.mouse_anchor
        elif g.get_mouse_button(window,g.MOUSE_BUTTON_RIGHT)==g.PRESS:
            _,h=g.get_window_size(window)
            action=mujoco.mjtMouse.mjMOUSE_MOVE_V if (g.get_key(window,g.KEY_LEFT_SHIFT)==g.PRESS) else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(self.sim.model,action,dx/h,dy/h,self.scene,self.cam)

    def scroll(self,window,x,y):
        if not self.dragging:
            mujoco.mjv_moveCamera(self.sim.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,self.scene,self.cam)

    def key(self,window,key,scancode,action,mods):
        g=self.glfw
        if action!=g.PRESS: return
        if key==g.KEY_ESCAPE: g.set_window_should_close(window,True)
        if key==g.KEY_SPACE: self.paused=not self.paused
        if key==g.KEY_F: self.follow=not self.follow
        if key==g.KEY_R:
            self.sim.reset(); self.selected=None; self.dragging=False

    def desired_force(self):
        desired=np.zeros((9,3))
        if self.dragging:
            i=self.selected; bid=self.sim.points[i]
            vel=np.zeros(6)
            mujoco.mj_objectVelocity(self.sim.model,self.sim.data,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
            desired[i]=80*(self.goal-self.sim.data.xpos[bid])-4*vel[3:]
        return desired

    def render(self):
        s=self.sim
        if self.follow and not self.dragging: self.cam.lookat[:]=s.data.qpos[:3]
        self.update_scene()
        for i,force in enumerate(s.forces):
            if np.linalg.norm(force)<.05: continue
            start=s.data.xpos[s.points[i]]
            geom=self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.35,.05,1],dtype=np.float32))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.012,start,start+.02*force)
            self.scene.ngeom+=1
        if self.selected is not None:
            geom=self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.025]*3),s.data.xpos[s.points[self.selected]],np.eye(3).ravel(),np.array([.1,1,.9,1],dtype=np.float32))
            self.scene.ngeom+=1
        w,h=self.glfw.get_framebuffer_size(self.window); viewport=mujoco.MjrRect(0,0,w,h)
        mujoco.mjr_render(viewport,self.scene,self.context)
        magnitude=float(np.linalg.norm(s.forces,axis=1).sum())
        name=POINT_NAMES[self.selected] if self.selected is not None else 'none'
        status=f"PURE STUDENT  t={s.data.time:.2f}s  {'PAUSED' if self.paused else 'RUNNING'}\nPoint: {name} | force {magnitude:.1f} N (cap 30 N)\nRoot height {s.data.qpos[2]:.2f} m"
        if s.first_fall is not None: status+='\nFALL DETECTED -- R to reset'
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,status,'',self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,
            'Left drag upper body: force | release: remove force\nRight drag: orbit | Shift+right: pan | wheel: zoom\nR: standing reset | Space: pause | F: follow | Esc: quit','',self.context)
        self.glfw.swap_buffers(self.window)

    def run(self,frames=0,screenshot=None):
        previous=time.perf_counter(); accumulator=0.
        try:
            while not self.glfw.window_should_close(self.window):
                self.glfw.poll_events()
                now=time.perf_counter(); accumulator+=min(now-previous,.1); previous=now
                if self.paused: accumulator=0.
                while accumulator>=self.sim.model.opt.timestep:
                    self.sim.step(self.desired_force()); accumulator-=self.sim.model.opt.timestep
                self.render(); self.frame+=1
                if frames and self.frame>=frames:
                    if screenshot:
                        from PIL import Image
                        w,h=self.glfw.get_framebuffer_size(self.window)
                        rgb=np.zeros((h,w,3),dtype=np.uint8)
                        mujoco.mjr_readPixels(rgb,None,mujoco.MjrRect(0,0,w,h),self.context)
                        Image.fromarray(rgb[::-1]).save(screenshot)
                    break
        finally:
            self.context.free(); self.glfw.destroy_window(self.window); self.glfw.terminate()


def make_simulation(bundle=None, model_path=DEFAULT_MODEL, policy='direct'):
    bundle = bundle or (DEFAULT_BUNDLE if policy == 'direct' else ROOT / 'deploy/mujoco_whole_body' / policy)
    backend = json.loads((bundle/'metadata.json').read_text()).get('backend', 'direct')
    if backend in ('sonic','ceer','jacobian'):
        from mujoco_residual import ResidualSimulation
        return ResidualSimulation(bundle, model_path)
    return Simulation(bundle, model_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,help='Override the selected policy bundle')
    p.add_argument('--policy',choices=['direct','sonic','ceer','jacobian'],default='direct')
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--test-force',action='store_true',help='Headless 15 N hand pull during 5-7 s')
    p.add_argument('--report',type=Path)
    p.add_argument('--gui-frames',type=int,default=0,help=argparse.SUPPRESS)
    p.add_argument('--screenshot',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.headless and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        p.error('No DISPLAY: run this command in a graphical desktop terminal (or SSH with X forwarding). Use --headless for numerical checks.')
    s=make_simulation(args.bundle,args.model,args.policy)
    if not args.headless:
        Interactive(s).run(args.gui_frames,args.screenshot)
        return
    heights=[]; maxforce=0.; maxerror=0.; actions=[]
    for _ in range(round(args.seconds/s.model.opt.timestep)):
        desired=np.zeros((9,3))
        if args.test_force and 5<=s.data.time<7: desired[4]=[15,0,0]
        s.step(desired)
        heights.append(float(s.data.qpos[2])); maxforce=max(maxforce,float(np.linalg.norm(s.forces,axis=1).max()))
        if s.steps%s.decimation==1:
            expected=(s.forces@s.data.xmat[s.root].reshape(3,3)).ravel()
            # last_obs is before mj_step; verify through observation at the same state instead.
            maxerror=max(maxerror,float(np.max(np.abs(s.observation()[238:265]-expected))))
            actions.append(float(np.max(np.abs(s.last_action))))
    report={'seconds':args.seconds,'scripted_force':args.test_force,'root_height_min':min(heights),
        'root_height_final':heights[-1],'first_fall_time':s.first_fall,'max_force_N':maxforce,
        'force_observation_max_error':maxerror,'max_abs_action':max(actions),'finite':bool(np.isfinite(s.data.qpos).all()),
        'root_final_xyz':s.data.qpos[:3].tolist()}
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
