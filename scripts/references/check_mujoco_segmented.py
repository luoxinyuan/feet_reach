"""Portable 18-point observation, wrench and reset checks (no Isaac/data)."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from mujoco_whole_body_student import make_simulation

s=make_simulation(policy='segmented')
assert s.observation().shape==(408,) and s.forces.shape==(18,3)
initial=s.observation().copy()
errors=[]
# Each independent force channel, including offset points sharing the torso.
for i in range(18):
    s.reset()
    desired=np.zeros_like(s.forces);desired[i]=[3.,-2.,1.]
    points=s.point_positions().copy();com=s.data.xipos.copy()
    s.step(desired)
    force=np.sum(s.data.xfrc_applied[:,:3],axis=0)
    moment=np.sum(s.data.xfrc_applied[:,3:]+np.cross(com,s.data.xfrc_applied[:,:3]),axis=0)
    np.testing.assert_allclose(force,s.forces.sum(axis=0),atol=1e-12)
    np.testing.assert_allclose(moment,np.cross(points,s.forces).sum(axis=0),atol=1e-12)
    obs=s.observation();expected=(s.forces@s.data.xmat[s.root].reshape(3,3)).ravel()
    errors.append(float(np.max(np.abs(obs[238:292]-expected))))
    assert s.nearest_point(s.points[i],s.point_positions()[i])==i
# Shared-link forces must sum, not overwrite; compare generalized forces too.
s.reset();desired=np.zeros_like(s.forces);desired[[9,10,17]]=[[2,1,0],[-1,2,1],[1,0,2]]
points=s.point_positions().copy();qpos=s.data.qpos.copy();qvel=s.data.qvel.copy()
s.step(desired)
expected_data=mujoco.MjData(s.model);expected_data.qpos[:]=qpos;expected_data.qvel[:]=qvel
mujoco.mj_forward(s.model,expected_data)
qforce=np.zeros(s.model.nv)
for bid,point,force in zip(s.points,points,s.forces):
    mujoco.mj_applyFT(s.model,expected_data,force,np.zeros(3),point,int(bid),qforce)
expected_data.xfrc_applied[:]=s.data.xfrc_applied
mujoco.mj_forward(s.model,expected_data)
# mj_applyFT is also the MuJoCo Cartesian-wrench-to-generalized-force reference.
aggregate=np.zeros(s.model.nv)
for bid,wrench in enumerate(s.data.xfrc_applied):
    if bid:mujoco.mj_applyFT(s.model,expected_data,wrench[:3],wrench[3:],expected_data.xipos[bid],bid,aggregate)
np.testing.assert_allclose(aggregate,qforce,atol=1e-12)
s.reset();np.testing.assert_array_equal(s.observation(),initial)
assert max(errors)<1e-6
print(json.dumps(dict(points=18,observation_dim=408,force_observation_max_error=max(errors),point_picking=True,net_wrench=True,generalized_force=True,reset=True),indent=2))
