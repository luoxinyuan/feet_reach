"""Add a measured root-path inset to the recorded video; preserve every frame."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--policy-label');a=p.parse_args();meta=json.loads((a.directory/'summary.json').read_text())
trials=meta['trials'];data=[np.load(a.directory/(r['name']+'.npz')) for r in trials]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
reader=imageio.get_reader(a.directory/'rollout.mp4');writer=imageio.get_writer(a.directory/'interaction.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
frames_per_trial=(len(data[0]['root'])+1)//2
count=0
for i,frame in enumerate(reader):
 trial=i//frames_per_trial;step=min((i%frames_per_trial)*2,len(data[trial]['root'])-1);path=data[trial]['root'][:,0,:2].copy();path-=path[0];lo=path.min(0);hi=path.max(0);span=max(float((hi-lo).max())*1.15,1.);center=(hi+lo)/2
 def project(x):return np.array([1135+(x[0]-center[0])*210/span,592-(x[1]-center[1])*170/span])
 im=Image.fromarray(frame);draw=ImageDraw.Draw(im,'RGBA')
 if a.policy_label:
  draw.rectangle((0,0,1280,36),fill=(10,16,25,255))
  draw.text((15,10),f'{a.policy_label} | {trials[trial]["name"].upper()} | trial {trial+1}/{len(trials)} | {step*.02:.2f}/10.00 s',font=font,fill='white')
 draw.rectangle((0,87,1280,115),fill=(10,16,25,255));draw.text((15,91),'Independent standing reset per trial | +X forward, +Y left at start | no motion target | inset = measured path',font=font,fill=(110,230,255))
 draw.rectangle((990,470,1278,718),fill=(10,16,25,225));draw.text((1000,478),'ROOT PATH: +X right, +Y up',font=small,fill='white');draw.text((1000,697),f'plot width {span:.1f} m | cyan: measured',font=small,fill=(100,230,255))
 pixels=[tuple(project(x)) for x in path[:step+1]]
 if len(pixels)>1:draw.line(pixels,fill=(60,220,255),width=3)
 origin=project(path[0]);now=project(path[step]);draw.ellipse((origin[0]-4,origin[1]-4,origin[0]+4,origin[1]+4),fill='white');draw.ellipse((now[0]-5,now[1]-5,now[0]+5,now[1]+5),fill=(255,100,80))
 yaw=float(data[trial]['yaw'][step,0]);tip=now+np.array([np.cos(yaw),-np.sin(yaw)])*15;draw.line([tuple(now),tuple(tip)],fill='white',width=3)
 writer.append_data(np.asarray(im));count+=1
 if i==125:im.save(a.directory/'preview.png')
writer.close();reader.close();frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(a.directory/'interaction.mp4'));assert frames==meta['frames']==count
(a.directory/'video_verified.json').write_text(json.dumps({'frames':frames,'seconds':seconds,'resolution':[1280,720],'source_frames_preserved':True,'path':'interaction.mp4'},indent=2));print('VIDEO_VERIFIED',a.directory,frames,seconds,flush=True)
