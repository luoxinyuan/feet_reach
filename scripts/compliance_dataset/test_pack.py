import numpy as np
from scipy.spatial.transform import Rotation
from pack import normalized_qpos,derivatives
n=100;t=np.arange(n)/50;q=np.zeros((n,36),np.float32);q[:,:3]=np.c_[1+2*t,3-t,.75+0*t];q[:,3:7]=Rotation.from_euler('z',.4*t).as_quat()[:,[3,0,1,2]];q[:,7:]=t[:,None]*.2
q[30:60,3:7]*=-1
p=normalized_qpos(q);assert np.allclose(p[0,:2],0);assert np.all((p[1:,3:7]*p[:-1,3:7]).sum(-1)>0)
v,w,j=derivatives(p);assert np.allclose(v,[2,-1,0],atol=2e-5);assert np.allclose(w,[0,0,.4],atol=2e-5);assert np.allclose(j,.2,atol=2e-5)
# Independent static clip must have no preceding motion's artificial boundary velocity.
r=p[-10:].copy();r[:]=r[0];v,w,j=derivatives(r);assert max(abs(v).max(),abs(w).max(),abs(j).max())<1e-6
print('PASS: quaternion continuity, XY normalization, analytic velocities, isolated clip boundary')
