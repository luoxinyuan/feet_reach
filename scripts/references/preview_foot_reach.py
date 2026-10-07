"""Standalone animated USD-mesh preview of a generated reach trajectory."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
import plotly.graph_objects as go
import trimesh
from generate_wall_support import load_usd
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'artifacts/foot_reach'
r=json.loads((ROOT/'artifacts/wall_support/reference.json').read_text())
m,names,meshes,_=load_usd(ROOT/r['source_usd']);d=m.createData()
w=np.load(out/'workspace.npz')
i=int(w['trajectory_endpoint_ids'][np.argmax(np.linalg.norm(w['points_root'][w['trajectory_endpoint_ids']]-w['points_root'][0],axis=1))])
manifest=json.loads((out/'manifest.json').read_text())
clip=next(c for c in manifest['clips'] if c['endpoint_id']==i and c['variant']==0)
raw=np.load(out/clip['file'])
qs=np.c_[raw['root_pos'],raw['root_rot'],raw['dof_pos']]
geometry=[]
for name,v,f in meshes:
    mesh=trimesh.Trimesh(v,f,process=True)
    if len(mesh.faces)>500: mesh=mesh.simplify_quadric_decimation(face_count=500)
    geometry.append((name,m.getFrameId(name),np.asarray(mesh.vertices),np.asarray(mesh.faces)))
def traces(q):
    pin.framesForwardKinematics(m,d,q)
    result=[]
    for name,frame,v,f in geometry:
        v=v@d.oMf[frame].rotation.T+d.oMf[frame].translation
        c='#20b8a6' if name.startswith('left_') else '#f0a34c' if name.startswith('right_') else '#64748b'
        result.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=c,name=name,hoverinfo='name'))
    return result
fig=go.Figure(traces(qs[0]))
for v,color in [(np.array([[-.45,.25,0],[.55,.25,0],[.55,.25,1.5],[-.45,.25,1.5]]),'#94a3b8'),
                (np.array([[-.5,-.4,0],[.6,-.4,0],[.6,.25,0],[-.5,.25,0]]),'#cbd5e1')]:
    fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=[0,0],j=[1,2],k=[2,3],color=color,opacity=.25))
p=w['points_world'][w['trajectory_endpoint_ids']]
fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=2,color='#a855f7'),name='Validated targets'))
frames=[go.Frame(data=traces(qs[t]),traces=list(range(len(geometry))),name=str(t)) for t in range(0,len(qs),5)]
fig.frames=frames
fig.update_layout(title=f'G1 wall-supported left-foot reach | endpoint {i} | 5× frame stride',
    scene=dict(aspectmode='data',xaxis=dict(range=[-.45,.55]),yaxis=dict(range=[-.4,.35]),zaxis=dict(range=[0,1.5]),
    camera=dict(eye=dict(x=1.7,y=-2,z=.8))),
    updatemenus=[dict(type='buttons',buttons=[dict(label='Play',method='animate',args=[None,{'frame':{'duration':100,'redraw':True},'fromcurrent':True}]),
    dict(label='Pause',method='animate',args=[[None],{'mode':'immediate','frame':{'duration':0,'redraw':False}}])])],
    sliders=[dict(steps=[dict(label=f'{int(f.name)/50:.1f}s',method='animate',args=[[f.name],{'mode':'immediate','frame':{'duration':0,'redraw':True}}]) for f in frames])])
fig.write_html(out/'trajectory_preview.html',include_plotlyjs=True)
print(out/'trajectory_preview.html')
