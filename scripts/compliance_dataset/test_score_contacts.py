import torch,numpy as np
from score_contacts import Geometry,score
G=Geometry('cpu');torch.set_num_threads(2)
q=torch.zeros(2,36);q[:,2]=.75;q[:,3]=1;q[:,7:]=torch.rand(2,29)*.1
p,J=G.fk(q,True)
for joint in (12,15,19,22,28):
 eps=1e-4;b=q.clone();b[:,7+joint]+=eps
 assert ((G.fk(b)-p)/eps-J[:,:,:,6+joint-12]/.3).abs().max()<.003
# Root translation derivative and quaternion sign invariance.
b=q.clone();b[:,0]+=1e-3
assert ((G.fk(b)-p)/1e-3-J[:,:,:,0]/.1).abs().max()<.001
b=q.clone();b[:,3:7]*=-1
assert torch.allclose(G.fk(b),p,atol=1e-6)
r=score(G,q,q,q,q,{'residual':.65,'group_residual':.9})
assert set(r)=={1,2,3,4,5}
for k,v in r.items():assert np.isfinite(v).all() and (v[:,0]==1).all() and (v[:,3]==0).all()
print('PASS: six FK finite differences, quaternion sign, static passage for all contact counts')
