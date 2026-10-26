"""Export exact training USD joint frames for simulator-independent preprocessing."""
import json
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
from pxr import Usd, UsdPhysics
path=Path('active_adaptation/assets/G1/g1_flat_fullcol.usd')
stage=Usd.Stage.Open(str(path.resolve()))
def quat(q): return [q.GetReal(),*q.GetImaginary()]
edges=[]
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint): continue
    j=UsdPhysics.Joint(prim); a=j.GetBody0Rel().GetTargets(); b=j.GetBody1Rel().GetTargets()
    if not a or not b: continue
    axis=None
    if prim.IsA(UsdPhysics.RevoluteJoint):axis=UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
    elif not prim.IsA(UsdPhysics.FixedJoint):raise ValueError(prim)
    edges.append(dict(parent=a[0].name,child=b[0].name,name=prim.GetName(),axis=axis,
        p0=list(j.GetLocalPos0Attr().Get()),r0=quat(j.GetLocalRot0Attr().Get()),
        p1=list(j.GetLocalPos1Attr().Get()),r1=quat(j.GetLocalRot1Attr().Get())))
import hashlib
out=dict(usd=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),edges=edges)
Path('artifacts/ceer_segmented_jacobian/geometry.json').write_text(json.dumps(out,indent=2))
print('GEOMETRY_EXPORTED',len(edges),flush=True)
app.close()
