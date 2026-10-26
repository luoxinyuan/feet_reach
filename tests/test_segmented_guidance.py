"""Pure CPU tests for segmentation, identity-aware smoothness and cache intent."""
import numpy as np
from active_adaptation.utils.segmented_guidance import select_segments,optimize_forces


def test_left_right_left_segments():
    # Independent coordinates can only be explained at the matching point.
    a=np.zeros((150,2,6));a[:,0,0]=1;a[:,1,3]=1
    b=np.zeros((150,2));b[:50,0]=1;b[50:100,1]=1;b[100:,0]=1
    slots,stats=select_segments(a,b,2,point_cost=.1,switch_cost=.1)
    assert stats['segments']==3,(stats,slots)
    assert np.all(slots[:50,0]==0) and np.all(slots[50:100,0]==1) and np.all(slots[100:,0]==0)
    assert np.all(slots[:,1:]==-1)
    force,residual,info=optimize_forces(a,b,slots)
    dense=np.zeros((150,2,3))
    dense[np.arange(150),slots[:,0]]=force[:,0]
    assert np.linalg.norm(np.diff(dense,axis=0),axis=-1).max()<=3.00001
    assert np.linalg.norm(force,axis=-1).max()<=30.00001
    assert np.linalg.norm(force,axis=-1).sum(-1).max()<=60.00001
    assert not force[49].any() and not force[99].any()
    assert residual.mean()<.05,residual.mean()


def test_bilateral_requires_two_points():
    a=np.zeros((100,2,6));a[:,0,0]=1;a[:,1,3]=1;b=np.ones((100,2))
    slots,_=select_segments(a,b,2,point_cost=.05)
    assert np.all(slots[:,:2]==[0,1])


def test_stationary_no_force():
    a=np.ones((60,2,6));b=np.zeros((60,2))
    slots,_=select_segments(a,b,2)
    force,residual,_=optimize_forces(a,b,slots)
    assert (slots==-1).all() and not force.any() and not residual.any()

if __name__=='__main__':
    test_left_right_left_segments();test_bilateral_requires_two_points();test_stationary_no_force()
    print('SEGMENTED_GUIDANCE_TESTS_PASS')
