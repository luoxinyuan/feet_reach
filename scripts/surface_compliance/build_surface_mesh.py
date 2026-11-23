"""Export continuous triangle contact surfaces from shipped G1 STL geometry."""
import json,struct,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
xml=ROOT/'deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml';tree=ET.parse(xml).getroot();meshes={e.attrib['name']:e.attrib['file'] for e in tree.find('asset').findall('mesh')}
# Exposed main links; include head geometry attached to torso and hand coverings.
allowed={'pelvis','torso_link'}|{f'{s}_{b}' for s in ['left','right'] for b in ['hip_yaw_link','knee_link','ankle_roll_link','shoulder_roll_link','shoulder_yaw_link','elbow_link','wrist_roll_link','wrist_pitch_link','wrist_yaw_link']}
triangles=[];parts=[];hashes={};seen=set()
for body in tree.iter('body'):
 name=body.attrib['name']
 if name not in allowed:continue
 for g in body.findall('geom'):
  mesh=g.attrib.get('mesh')
  if mesh is None or mesh=='logo_link':continue
  key=(name,mesh,g.attrib.get('pos','0 0 0'),g.attrib.get('quat','1 0 0 0'))
  if key in seen:continue
  seen.add(key);path=xml.parent/'meshes'/meshes[mesh];raw=path.read_bytes();n=struct.unpack('<I',raw[80:84])[0]
  if len(raw)!=84+50*n:raise ValueError(f'Expected binary STL: {path}')
  rec=np.frombuffer(raw,offset=84,count=n,dtype=[('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]);v=rec['v'].copy()
  from scipy.spatial.transform import Rotation
  quat=np.fromstring(key[3],sep=' ');rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();v=v@rotation.T+np.fromstring(key[2],sep=' ')
  area=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=-1)*.5;keep=area>1e-10;v=v[keep];area=area[keep]
  # Surface samples are barycentric, never a finite bank of fixed points.
  start=sum(len(x) for x in triangles);triangles.append(v.astype(np.float32));parts.append(dict(body=name,mesh=mesh,start=start,end=start+len(v),area=float(area.sum())))
  hashes[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
out=ROOT/'artifacts/surface_compliance/geometry';out.mkdir(parents=True,exist_ok=True);np.save(out/'triangles.npy',np.concatenate(triangles))
(out/'manifest.json').write_text(json.dumps(dict(parts=parts,source=str(xml),source_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),mesh_sha256=hashes,semantics='grasp/attachment: signed 3D forces; not unilateral palm contact',note='Mesh surface, not guaranteed globally exposed or human-reachable; floor-height filter at selection. Head mesh is attached to torso.'),indent=2));print(out,len(np.concatenate(triangles)),len(parts))
