"""SOCP force allocation with an exact sum-of-Euclidean-norm budget."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'artifacts/offline_force_baseline/tooling'))
import clarabel
import numpy as np
from scipy import sparse

def solve_sequence(A,dq,budget,k=80.,point_cap=30.,slew=None,smooth=.01,error_slack_rad=1e-4,previous_force=None):
    """First minimize decoded-dq error; then minimize force within that error tolerance."""
    T,n,d=A.shape;m=d//3;nf=T*d;nt=T*m;nv=nf+nt
    C=sparse.block_diag(list(A/k*100),format='csc');y=(dq*100).ravel()
    # Variable x=[flattened physical forces, per-point norm epigraphs].
    ii=np.arange(nt);row=np.concatenate([4*ii,*(4*ii+j+1 for j in range(3))]);col=np.concatenate([nf+ii,*(3*ii+j for j in range(3))])
    point=sparse.coo_matrix((-np.ones(len(row)),(row,col)),shape=(4*nt,nv)).tocsc()
    blocks=[point];bs=[np.zeros(4*nt)];cones=[clarabel.SecondOrderConeT(4) for _ in range(nt)]
    zmap=sparse.hstack([sparse.csc_matrix((nt,nf)),sparse.eye(nt)],format='csc')
    total=sparse.kron(sparse.eye(T),np.ones((1,m)),format='csc')@zmap
    blocks += [zmap,total];bs += [np.full(nt,point_cap),np.full(T,budget)];cones += [clarabel.NonnegativeConeT(nt),clarabel.NonnegativeConeT(T)]
    D=sparse.kron(sparse.diags([-np.ones(T-1),np.ones(T-1)],[0,1],shape=(T-1,T)),sparse.eye(d),format='csc')
    if slew is not None:
        nr=(T-1)*m;coo=D.tocoo();r=4*(coo.row//3)+1+coo.row%3
        rate=sparse.coo_matrix((-coo.data,(r,coo.col)),shape=(4*nr,nv)).tocsc();b=np.zeros(4*nr);b[::4]=slew*.02
        blocks.append(rate);bs.append(b);cones += [clarabel.SecondOrderConeT(4) for _ in range(nr)]
    if previous_force is not None:
        if slew is None:raise ValueError('previous_force requires a slew limit')
        previous_force=np.asarray(previous_force,dtype=float).reshape(m,3)
        rows=(4*np.arange(m)[:,None]+1+np.arange(3)).ravel()
        initial=sparse.coo_matrix((-np.ones(d),(rows,np.arange(d))),shape=(4*m,nv)).tocsc()
        initial_b=np.zeros((m,4));initial_b[:,0]=slew*.02;initial_b[:,1:]=-previous_force
        blocks.append(initial);bs.append(initial_b.ravel());cones += [clarabel.SecondOrderConeT(4) for _ in range(m)]
    con=sparse.vstack(blocks,format='csc');bound=np.concatenate(bs)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=150;settings.tol_gap_abs=1e-8;settings.tol_feas=1e-8;settings.tol_gap_rel=1e-8
    def run(P,q,aa,bb,cc):
        attempts=[]
        for tolerance in (1e-8, 1e-7, 1e-6):
            settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance;settings.tol_feas=tolerance
            result=clarabel.DefaultSolver(sparse.triu(P).tocsc(),q,aa,bb,cc,settings).solve()
            attempts.append(str(result.status))
            if str(result.status) in ('Solved','AlmostSolved'):break
        else:raise RuntimeError(str(attempts))
        return np.asarray(result.x),dict(status=str(result.status),iterations=result.iterations,time_s=result.solve_time,tolerance=tolerance,attempts=attempts)
    P=sparse.block_diag([2*(C.T@C)+1e-10*sparse.eye(nf),sparse.csc_matrix((nt,nt))],format='csc')
    x1,s1=run(P,np.r_[-2*(C.T@y),np.zeros(nt)],con,bound,cones)
    optimum=np.linalg.norm(C@x1[:nf]-y);allow=optimum+100*error_slack_rad*np.sqrt(T*n)
    error_constraint=sparse.vstack([sparse.csc_matrix((1,nv)),sparse.hstack([-C,sparse.csc_matrix((T*n,nt))])],format='csc')
    con2=sparse.vstack([con,error_constraint],format='csc');bound2=np.r_[bound,allow,-y]
    h=1e-8*sparse.eye(nf)+(smooth/T)*(D.T@D)
    P2=sparse.block_diag([2*h,sparse.csc_matrix((nt,nt))],format='csc')
    q2=np.r_[np.zeros(nf),np.full(nt,1/T)]
    if previous_force is not None:
        diagonal=np.zeros(nv);diagonal[:d]=2*smooth/T
        P2=P2+sparse.diags(diagonal,format='csc');q2[:d]=-2*smooth/T*previous_force.ravel()
    x,s2=run(P2,q2,con2,bound2,cones+[clarabel.SecondOrderConeT(1+T*n)])
    f=x[:nf].reshape(T,m,3);norms=np.linalg.norm(f,axis=-1)
    violation=dict(total_N=float(max(0,norms.sum(1).max()-budget)),point_N=float(max(0,norms.max()-point_cap)),
                   slew_N_s=float(max(0,np.linalg.norm(np.diff(f,axis=0),axis=-1).max(initial=0)/.02-slew)) if slew is not None else 0.,
                   error_bound=float(max(0,np.linalg.norm(C@x[:nf]-y)-allow)))
    if previous_force is not None:
        violation['slew_N_s']=max(violation['slew_N_s'],float(max(0,np.linalg.norm(f[0]-previous_force,axis=-1).max()/.02-slew)))
    if violation['total_N']>1e-4 or violation['point_N']>1e-4 or violation['slew_N_s']>1e-3 or violation['error_bound']>1e-4:raise RuntimeError(violation)
    return f,dict(stage1=s1,stage2=s2,best_rmse_rad=optimum/100/np.sqrt(T*n),error_slack_rad=error_slack_rad,violations=violation)
