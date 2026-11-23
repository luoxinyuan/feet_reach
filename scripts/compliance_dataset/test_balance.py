import numpy as np
from balance import features,IntervalBudget,select

def example():
 q=np.zeros((1,17,36),np.float32);q[:,:,2]=.75;q[:,:,3]=1
 pos=np.zeros((1,17,4,3),np.float32);pos[:,:,:2]=np.array([[[.2,.3,.8],[.2,-.3,.8]]]);pos[:,:,2]=[0,0,.8];pos[:,:,3]=[0,0,1.8]
 return q,pos
q,p=example();r=features(q,p,np.array([4.]),np.array([True]));assert r['labels']==['still']
# Root translation should not become hand articulation.
q,p=example();dx=np.linspace(0,1,17);q[0,:,0]+=dx;p[0,:,:,0]+=dx[:,None]
r=features(q,p,np.array([4.]),np.array([False]));assert r['labels']==['locomotion'];assert not any('arm' in x for x in r['auxiliary'][0]);assert r['subtypes']==['forward_sustained']
# Left hand reaches forward with corresponding arm joint movement.
q,p=example();p[0,:,0,0]+=np.linspace(0,.3,17);q[0,:,22]=np.linspace(0,.5,17)
r=features(q,p,np.array([4.]),np.array([False]));assert r['labels']==['single_arm'] and r['subtypes']==['left_forward']
# Heading change must not appear as relative hand movement.
q,p=example();yaw=np.linspace(0,.7,17);q[0,:,3]=np.cos(yaw/2);q[0,:,6]=np.sin(yaw/2)
for t,a in enumerate(yaw):
 x=p[0,t,:,0].copy();y=p[0,t,:,1].copy();p[0,t,:,0]=np.cos(a)*x-np.sin(a)*y;p[0,t,:,1]=np.sin(a)*x+np.cos(a)*y
r=features(q,p,np.array([4.]),np.array([False]));assert r['labels']==['yaw'] and r['subtypes']==['left'];assert not any('arm' in x for x in r['auxiliary'][0])
# No overlap, no duplicate signature, no per-clip overrun, adjacency is valid.
b=IntervalBudget(400);row=dict(dataset='a',clip_id=0,start=0,end=200,fingerprint='x')
assert b.accept(row,400)
assert not b.accept({**row,'start':100,'end':300,'fingerprint':'y'},400)
assert not b.accept({**row,'clip_id':1},400)
assert b.accept({**row,'start':200,'end':400,'fingerprint':'z'},400)
assert not b.accept({**row,'start':400,'end':600,'fingerprint':'w'},400)
records=[dict(dataset='a',clip_id=i,start=0,end=200,fingerprint=str(i),primary='still',subtype='upright',diversity=[0]) for i in range(5)]
chosen,counts,*_=select(records,{'still':.002},123,60)
assert counts['still']==200 and len(chosen)==1
print('PASS: root-relative labels, direction, yaw invariance, overlap, dedup, caps, budget')
