"""Requires optional Playwright + Chromium; neither is needed by the web server."""
import argparse,json,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser(description='Browser integration check for the MuJoCo viewer')
parser.add_argument('--url',default='http://127.0.0.1:8765')
base=parser.parse_args().url.rstrip('/')
out=Path(__file__).resolve().parents[2]/'artifacts/mujoco_whole_body_student/web';out.mkdir(parents=True,exist_ok=True)
def state():return json.load(urllib.request.urlopen(base+'/state'))
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1280,'height':1100})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(base)
 page.wait_for_function("document.querySelector('#toggle').disabled===false")
 page.wait_for_timeout(1000)
 print('CONNECTED',state())
 page.screenshot(path=str(out/'standing.png'),full_page=True)
 # Find a visible torso point while paused; mouse coordinates must match scaled canvas.
 box=page.locator('#view').bounding_box()
 selected=False
 for ny in [.43,.46,.49,.52]:
  for nx in [.50,.48,.52,.46,.54]:
   x=box['x']+box['width']*nx;y=box['y']+box['height']*ny
   page.mouse.move(x,y);page.mouse.down();page.wait_for_timeout(100)
   if state()['dragging']:
    selected=True;break
   page.mouse.up()
  if selected:break
 assert selected,'failed to pick body'
 selected_name=state()['selected'];page.mouse.up()
 page.locator('#toggle').click();page.wait_for_timeout(150)
 # Pause/reset to put robot back at exact initial location, then run while dragging.
 page.locator('#toggle').click();page.wait_for_timeout(150)
 page.locator('#reset').click();page.wait_for_timeout(150)
 # Starting simulation via keyboard without releasing pointer.
 page.mouse.move(x,y);page.mouse.down();page.wait_for_timeout(100)
 assert state()['dragging']
 page.evaluate("post({type:'pause',paused:false})")
 page.mouse.move(x+220,y-35,steps=10)
 page.wait_for_timeout(500)
 force_state=state();print('FORCE',force_state)
 assert force_state['force_N']>1 and force_state['force_N']<=30.000001
 page.screenshot(path=str(out/'drag.png'),full_page=True)
 page.mouse.up();page.wait_for_timeout(500)
 released=state();print('RELEASE',released)
 assert released['force_N']<1e-8 and not released['dragging']
 page.wait_for_timeout(1500)
 speed=state()['realtime_factor'];assert speed>.75,speed
 page.keyboard.press('Space');page.wait_for_timeout(200);assert state()['paused']
 page.keyboard.press('r');page.wait_for_timeout(200)
 assert state()['time']==0 and abs(state()['root_height']-.74)<1e-8
 # Browser controls on a resized canvas use normalized coordinates.
 page.set_viewport_size({'width':850,'height':900});page.wait_for_timeout(150)
 box=page.locator('#view').bounding_box()
 page.mouse.move(box['x']+box['width']*nx,box['y']+box['height']*ny)
 page.mouse.down();page.wait_for_timeout(150);assert state()['dragging']
 page.mouse.up()
 # A second independent client cannot change simulation while first owns it.
 token=page.evaluate('token')
 req=urllib.request.Request(base+'/command',data=json.dumps({'type':'reset','client':'second-client'}).encode(),headers={'X-Control-Token':token})
 try:urllib.request.urlopen(req);raise AssertionError('second controller accepted')
 except urllib.error.HTTPError as e:assert e.code==409
 page.locator('#toggle').click();page.wait_for_timeout(200)
 page.close();time.sleep(3.5)
 disconnected=state();assert disconnected['paused'] and disconnected['force_N']==0
 browser.close()
 assert not errors,errors
 result={'browser_errors':errors,'selected':selected_name,'drag_force_N':force_state['force_N'],
  'release_force_N':released['force_N'],'realtime_factor':speed,'resized_picking':True,
  'reset':True,'second_controller_rejected':True,'disconnect_paused':True}
 (out/'browser_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
