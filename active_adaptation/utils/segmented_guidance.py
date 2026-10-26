"""Offline reference-only sparse point selection and constrained trajectory forces.

No inverse dynamics or reference access from the student. NumPy/SciPy implementation
uses the exact training USD joint geometry exported once by Isaac Sim.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import sparse
import osqp


def rot(q):
    q=np.asarray(q); return Rotation.from_quat(q[..., [1,2,3,0]]).as_matrix()


def geometry_jacobians(q, root_pos, root_quat, joint_names, points, geometry):
    """Return world point positions and [T,P,3,6+J] link-origin Jacobians."""
    n=len(q); eye=np.broadcast_to(np.eye(3),(n,3,3))
    states={'pelvis':(root_pos,rot(root_quat),[])}; joints={}; pending=list(geometry['edges'])
    while pending:
        ready=[e for e in pending if e['parent'] in states]
        if not ready: raise ValueError('Disconnected geometry')
        for e in ready:
            p,r,anc=states[e['parent']]; jr=r@rot(e['r0']); jp=p+np.einsum('tij,j->ti',r,e['p0'])
            if e['axis'] is not None:
                idx=joint_names.index(e['name']); axis=np.eye(3)['XYZ'.index(e['axis'])]
                joints[idx]=(jp,np.einsum('tij,j->ti',jr,axis))
                rr=Rotation.from_rotvec(q[:,idx,None]*axis).as_matrix(); anc=anc+[idx]
            else: rr=eye
            cr=jr@rr@rot(e['r1']).T
            cp=jp-np.einsum('tij,j->ti',cr,e['p1'])
            states[e['child']]=(cp,cr,anc);pending.remove(e)
    positions=[];jac=[]
    for point in points:
        p,r,anc=states[point['body']]; p=p+np.einsum('tij,j->ti',r,point['offset'])
        j=np.zeros((n,3,6+len(joint_names))); j[:,:,:3]=eye
        j[:,:,3:6]=np.cross(np.eye(3)[None,:,:],(p-root_pos)[:,None,:]).transpose(0,2,1)
        for idx in anc:
            jp,axis=joints[idx];j[:,:,6+idx]=np.cross(axis,p-jp)
        positions.append(p);jac.append(j)
    return np.stack(positions,1),np.stack(jac,1)


def nominal_target(q, root_pos, root_quat, dt=.02):
    dq=np.diff(q,axis=0,append=q[-1:]);dp=np.diff(root_pos,axis=0,append=root_pos[-1:])
    r=Rotation.from_matrix(rot(root_quat)); rv=(r[1:]*r[:-1].inv()).as_rotvec()
    rv=np.concatenate([rv,np.zeros((1,3))])
    target=np.zeros((len(q),6+q.shape[1]));target[:,:3]=(60+8/dt)*dp
    norms=np.linalg.norm(target[:,:3],axis=-1,keepdims=True)
    target[:,:3]*=np.minimum(1,40/np.maximum(norms,1e-9))
    target[:,5]=np.clip((8+1.5/dt)*rv[:,2],-4,4)
    target[:,6:]=80*dq
    return target


def weighted_problem(jac,target,names):
    # Exclude unactuated-by-upper-body leg columns. Keep stationary upper joints:
    # cross-talk into an otherwise still arm must be penalized as well.
    groups=[list(range(3)),list(range(3,6))]
    groups += [[6+i for i,n in enumerate(names) if n.startswith(prefix)]
               for prefix in ('waist_','left_shoulder','right_shoulder')]
    groups[3]=[6+i for i,n in enumerate(names) if n.startswith('left_') and not any(x in n for x in ('hip','knee','ankle'))]
    groups[4]=[6+i for i,n in enumerate(names) if n.startswith('right_') and not any(x in n for x in ('hip','knee','ankle'))]
    cols=sum(groups,[]); weights=np.zeros_like(target)
    for ids,floor in zip(groups,(3.,.3,.3,.3,.3)):
        scale=np.maximum(np.linalg.norm(target[:,ids],axis=1),floor)
        weights[:,ids]=1/scale[:,None]
    a=jac.transpose(0,3,1,2).reshape(len(jac),target.shape[1],-1)
    return a[:,cols]*weights[:,cols,None], target[:,cols]*weights[:,cols], groups,weights


def fit_candidates(a,b,sets,penalty=.002):
    """Small ridge fits, conservatively cap L1 budgets before scoring."""
    costs=np.zeros((len(a),len(sets)))
    for size in (0,1,2,3):
        ids=[i for i,s in enumerate(sets) if len(s)==size]
        if not ids:continue
        if size==0:costs[:,ids]=np.sum(b*b,axis=-1)[:,None];continue
        cols=np.array([[3*p+k for p in sets[i] for k in range(3)] for i in ids])
        aa=a[:,:,cols].transpose(0,2,1,3)
        h=aa.transpose(0,1,3,2)@aa+penalty*np.eye(3*size)
        rhs=np.einsum('tcdi,td->tci',aa,b)
        f=np.linalg.solve(h,rhs[...,None])[...,0].reshape(len(a),len(ids),size,3)
        l1=np.abs(f).sum(-1)
        scale=np.minimum(1,np.minimum(30/np.maximum(l1.max(-1),1e-9),60/np.maximum(l1.sum(-1),1e-9)))
        f*=scale[...,None,None];f=f.reshape(len(a),len(ids),-1)
        err=np.einsum('tcdi,tci->tcd',aa,f)-b[:,None]
        costs[:,ids]=np.sum(err*err,axis=-1)+penalty*np.sum(f*f,axis=-1)
    return costs


def select_segments(a,b,npoints,block=10,point_cost=.035,switch_cost=.35):
    # Screen all singleton/pair layouts and greedily expand the best pairs to
    # triples at 10 Hz. This is a candidate search, not a global-optimum claim.
    sample=np.arange(0,len(a),5)
    base=[()]+list(itertools.combinations(range(npoints),1))+list(itertools.combinations(range(npoints),2))
    c=fit_candidates(a[sample],b[sample],base)
    pairs=np.array([i for i,s in enumerate(base) if len(s)==2])
    top=pairs[np.argsort(c[:,pairs],axis=1)[:,:3]]
    triples={tuple(sorted((*base[i],p))) for i in np.unique(top) for p in range(npoints) if p not in base[i]}
    # Retain top candidates at every sampled frame, including lower point counts.
    chosen={0}
    for size in (1,2):
        ids=np.array([i for i,s in enumerate(base) if len(s)==size])
        chosen.update(ids[np.argsort(c[:,ids],axis=1)[:,:3]].ravel().tolist())
    sets=[base[i] for i in sorted(chosen)]+sorted(triples)
    c=fit_candidates(a[sample],b[sample],sets)
    # Trim triples to those useful anywhere; preserve every winning sparse layout.
    keep=set(np.argsort(c+point_cost*np.array([len(s) for s in sets])[None],axis=1)[:,:8].ravel())|{0}
    sets=[sets[i] for i in sorted(keep)];c=c[:,sorted(keep)]
    starts=np.arange(0,len(a),block);em=[]
    for st in starts:
        mask=(sample>=st)&(sample<st+block)
        em.append(c[mask].mean(0)+point_cost*np.array([len(s) for s in sets]))
    em=np.array(em)
    changes=np.array([[len(set(s)^set(t)) for t in sets] for s in sets])
    transition=switch_cost*changes
    score=em[0].copy();parents=[]
    for e in em[1:]:
        z=score[:,None]+transition;par=z.argmin(0);parents.append(par)
        score=z[par,np.arange(len(sets))]+e
    path=[int(score.argmin())]
    for par in parents[::-1]:path.append(int(par[path[-1]]))
    path=path[::-1];slots=np.full((len(a),3),-1,np.int16)
    for st,state in zip(starts,path):slots[st:st+block,:len(sets[state])]=sets[state]
    return slots,dict(candidate_states=len(sets),segments=1+int(np.any(slots[1:]!=slots[:-1],axis=1).sum()))


def optimize_forces(a,b,slots,dt=.02,smooth=.04):
    """Sparse convex QP. L1 force balls conservatively imply Euclidean caps.

    Constraints are on physical point identities, not interchangeable slots.
    A zero-force frame before every layout switch prevents >3 simultaneous
    points during continuous physics interpolation.
    """
    t=len(slots);n=9*t; p=a.shape[-1]//3
    cols=np.maximum(slots,0)[...,None]*3+np.arange(3)
    aa=np.take_along_axis(a,cols.reshape(t,1,9),axis=2)
    aa*=np.repeat(slots>=0,3,axis=1)[:,None]
    big=sparse.block_diag(list(aa),format='csc')
    rows=[];cc=[];vals=[]
    for ti in range(t):
        for k,point in enumerate(slots[ti]):
            if point<0:continue
            for d in range(3):
                rows.append(ti*p*3+point*3+d);cc.append(ti*9+k*3+d);vals.append(1.)
    mapping=sparse.coo_matrix((vals,(rows,cc)),shape=(t*p*3,n)).tocsc()
    diff=sparse.diags([-np.ones(t-1),np.ones(t-1)],[0,1],shape=(t-1,t),format='csc')
    dmat=sparse.kron(diff,sparse.eye(p*3),format='csc')@mapping
    h=big.T@big+.002*sparse.eye(n)+smooth*(dmat.T@dmat)
    # Variables [force, absolute-force upper bound].
    P=sparse.block_diag((2*h,sparse.csc_matrix((n,n))),format='csc')
    q=np.r_[-2*(big.T@b.ravel()),np.full(n,1e-7)]
    eye=sparse.eye(n,format='csc');zero=sparse.csc_matrix((n,n))
    per=sparse.kron(sparse.eye(t*3),np.ones((1,3)),format='csc')
    total=sparse.kron(sparse.eye(t),np.ones((1,9)),format='csc')
    bound=np.where(np.repeat(slots>=0,3,axis=1).ravel(),30.,0.)
    switch=np.where(np.any(slots[1:]!=slots[:-1],axis=1))[0]
    bound.reshape(t,9)[switch]=0
    bound[:9]=0;bound[-9:]=0
    C=sparse.vstack([sparse.hstack([eye,zero]),sparse.hstack([eye,-eye]),sparse.hstack([-eye,-eye]),
                     sparse.hstack([zero,eye]),sparse.hstack([sparse.csc_matrix((t*3,n)),per]),
                     sparse.hstack([sparse.csc_matrix((t,n)),total]),
                     sparse.hstack([dmat,sparse.csc_matrix((dmat.shape[0],n))])],format='csc')
    slew=150*dt/np.sqrt(3)
    lo=np.r_[-bound,np.full(2*n,-np.inf),np.zeros(n),np.full(t*4,-np.inf),np.full(dmat.shape[0],-slew)]
    hi=np.r_[bound,np.zeros(2*n),np.full(n,np.inf),np.full(t*3,30.),np.full(t,60.),np.full(dmat.shape[0],slew)]
    solver=osqp.OSQP();solver.setup(P=sparse.triu(P).tocsc(),q=q,A=C,l=lo,u=hi,
        verbose=False,eps_abs=2e-4,eps_rel=2e-4,max_iter=4000,polish=False)
    result=solver.solve()
    if result.info.status_val not in (1,2):raise RuntimeError(f'Force QP: {result.info.status}')
    f=result.x[:n].reshape(t,3,3);f[np.abs(f)<1e-5]=0;f[slots<0]=0
    f[switch]=0;f[0]=0;f[-1]=0
    # Uniform numerical safety scaling preserves all homogeneous constraints.
    dense=np.asarray(mapping@f.ravel()).reshape(t,p,3)
    factor=min(1.,30/max(np.linalg.norm(f,axis=-1).max(),1e-9),
               60/max(np.linalg.norm(f,axis=-1).sum(-1).max(),1e-9),
               150*dt/max(np.linalg.norm(np.diff(dense,axis=0),axis=-1).max(initial=0),1e-9))
    f*=factor
    predicted=np.einsum('tdi,ti->td',aa,f.reshape(t,9))
    residual=np.linalg.norm(predicted-b,axis=-1)/np.maximum(np.linalg.norm(b,axis=-1),.2)
    return f.astype(np.float32),residual,dict(qp_iterations=int(result.info.iter),numerical_scale=float(factor))
