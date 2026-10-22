"""Exercise picking, dragging, force release and reset (run under xvfb-run)."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mujoco_whole_body_student import *
(ROOT/'artifacts/mujoco_whole_body_student').mkdir(parents=True,exist_ok=True)
s=Simulation(DEFAULT_BUNDLE,DEFAULT_MODEL)
u=Interactive(s);g=u.glfw
u.render()
# Select a visible upper-body pixel using the actual GLFW callback and mjv_select.
for y in range(280,510,8):
 for x in range(540,740,8):
  g.set_cursor_pos(u.window,x,y)
  u.mouse_button(u.window,g.MOUSE_BUTTON_LEFT,g.PRESS,0)
  if u.dragging:break
 if u.dragging:break
assert u.dragging,'No upper body could be selected'
selected=POINT_NAMES[u.selected]
u.mouse_move(u.window,x+400,y-50)
assert np.linalg.norm(u.desired_force())>0
for _ in range(150):s.step(u.desired_force())
u.render()
peak=float(np.linalg.norm(s.forces,axis=1).max())
assert abs(peak-30)<1e-6, 'Large mouse drags must saturate at 30 N'
assert np.allclose(s.data.xfrc_applied[s.points,:3],s.forces)
# Save force arrow preview before releasing.
from PIL import Image
w,h=g.get_framebuffer_size(u.window);rgb=np.zeros((h,w,3),dtype=np.uint8)
mujoco.mjr_readPixels(rgb,None,mujoco.MjrRect(0,0,w,h),u.context)
Image.fromarray(rgb[::-1]).save(ROOT/'artifacts/mujoco_whole_body_student/mouse_force_preview.png')
u.mouse_button(u.window,g.MOUSE_BUTTON_LEFT,g.RELEASE,0)
assert not u.dragging
for _ in range(150):s.step(u.desired_force())
assert np.linalg.norm(s.forces)<1e-8
u.key(u.window,g.KEY_R,0,g.PRESS,0)
assert s.steps==0 and np.allclose(s.data.qpos[s.qadr],s.q0)
result={'mouse_selected_point':selected,'mouse_force_peak_N':peak,'release_force_N':float(np.linalg.norm(s.forces)),'reset_passed':True,'render_passed':True}
print(json.dumps(result,indent=2));(ROOT/'artifacts/mujoco_whole_body_student/gui_test.json').write_text(json.dumps(result,indent=2))
u.context.free();g.destroy_window(u.window);g.terminate()
