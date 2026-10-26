"""World-frame point-force allocation and contact inverse dynamics (NumPy/SciPy).
Wrenches are [Fx,Fy,Fz,Mx,My,Mz], moments about the instantaneous COM.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def grasp_matrix(points, com):
    return np.concatenate([np.vstack((np.eye(3),skew(p-com))) for p in points],axis=1)


def feasible_yaw_wrench(points, com, force, yaw_moment):
    """Preserve the desired force and total COM yaw moment with two pure forces.

    Retain the physically induced roll/pitch moments rather than silently
    asking two point forces to realize an arbitrary six-dimensional wrench.
    """
    d=points[0]-points[1];n=np.linalg.norm(d)
    if n<.08: raise ValueError('hands too close')
    d=d/n;z=np.array([0.,0.,1.]);axis=z-d*d[2]
    if axis[2]<.2: raise ValueError('hand baseline nearly vertical: weak yaw authority')
    transport=np.cross(points.mean(0)-com,force)
    couple=axis*((yaw_moment-transport[2])/axis[2])
    return np.r_[force,transport+couple]


def guidance_wrench(com_velocity,com_acceleration,yaw_velocity,yaw_acceleration,
                    virtual_mass=4.,linear_drag=35.,virtual_yaw_inertia=.5,yaw_drag=6.):
    # Horizontal driving signal stays informative at constant speed. Ground
    # support, gravity and actual robot inertia are handled separately below.
    force=np.zeros(3);force[:2]=virtual_mass*com_acceleration[:2]+linear_drag*com_velocity[:2]
    return np.r_[force,0.,0.,virtual_yaw_inertia*yaw_acceleration+yaw_drag*yaw_velocity]


def allocate_contact_wrench(h, hand_jacobians, foot_jacobians, contact_mask,
                            points, com, desired, effort_limits, previous=None,
                            force_limit=60., friction=.7, exact=True, foot_yaws=None, max_force_rate=None, time_step=None):
    """Convex quadratic objective with linear constraints, solved by SLSQP.

    Variables: two world hand point forces + two world sole 6D contact wrenches.
    Hard floating-base equilibrium and actuator effort bounds, conservative
    pyramidal friction/CoP constraints; inactive feet carry exactly zero load.
    A per-axis force box Fmax/sqrt(3) conservatively enforces ||f_hand||<=Fmax.
    """
    A=grasp_matrix(points,com)
    H=np.concatenate([j[:3].T for j in hand_jacobians],axis=1)
    C=np.concatenate([j.T for j in foot_jacobians],axis=1)
    J=np.concatenate([H,C],axis=1)
    equality=J[:6];rhs=h[:6]
    task_rows = np.array([0,1,5]) if exact == "planar" else np.arange(6)
    if exact:
        equality=np.vstack((equality,np.c_[A[task_rows],np.zeros((len(task_rows),12))]));rhs=np.r_[rhs,desired[task_rows]]
    # Drop dependent equality rows by QR; reject inconsistent desired wrenches.
    from scipy.linalg import qr
    _,R,piv=qr(equality.T,pivoting=True,mode='economic')
    rank=np.linalg.matrix_rank(equality)
    idx=piv[:rank];eq=equality[idx];b=rhs[idx]
    if np.linalg.norm(equality@np.linalg.lstsq(equality,rhs,rcond=None)[0]-rhs)>1e-6:
        return {'success':False,'reason':'inconsistent_wrench'}
    rows=[];upper=[]
    for foot in range(2):
        offset=6+foot*6
        yaw=0. if foot_yaws is None else foot_yaws[foot]
        c,s=np.cos(yaw),np.sin(yaw)
        basis=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        for sx in (-1.,1.):
            for sy in (-1.,1.):
                r=np.zeros(18);r[offset:offset+3]=basis@np.array([sx,sy,-friction]);rows.append(r);upper.append(0.)
        for axis,lever in ((3,.025),(4,.045),(5,.015)):
            for sign in (-1.,1.):
                r=np.zeros(18);r[offset+3:offset+6]=sign*basis[:,axis-3];r[offset+2]=-lever;rows.append(r);upper.append(0.)
    # tau=h_joint-J_joint*x, with 15% effort headroom.
    rows.extend(J[6:]);upper.extend(h[6:]+.85*effort_limits)
    rows.extend(-J[6:]);upper.extend(.85*effort_limits-h[6:])
    U=np.array(rows);ub=np.array(upper)
    lo=np.full(18,-np.inf);hi=np.full(18,np.inf)
    lo[:6]=-force_limit/np.sqrt(3);hi[:6]=force_limit/np.sqrt(3)
    if previous is not None and max_force_rate is not None:
        if time_step is None or time_step <= 0: raise ValueError("positive time_step required for slew limit")
        delta=max_force_rate*time_step/np.sqrt(3)
        lo[:6]=np.maximum(lo[:6],np.asarray(previous)-delta)
        hi[:6]=np.minimum(hi[:6],np.asarray(previous)+delta)
    for i,active in enumerate(contact_mask):
        if active:lo[6+i*6+2]=0.;hi[6+i*6+2]=800.
        else:lo[6+i*6:12+i*6]=hi[6+i*6:12+i*6]=0.
    seed=linprog(np.zeros(18),A_ub=U,b_ub=ub,A_eq=eq,b_eq=b,bounds=list(zip(lo,hi)),method='highs')
    if not seed.success:return {'success':False,'reason':'contact_effort_or_force_infeasible'}
    # Normalize N and Nm; proximity only matters in relaxed diagnostic mode.
    W=np.diag([1/20]*3+[1/5]*3);B=np.c_[W@A,np.zeros((6,12))];target=W@desired
    Q=B.T@B+np.diag([2e-4]*6+[1e-7]*12);linear=-B.T@target
    if previous is not None:
        Q[:6,:6]+=np.eye(6)*1e-4;linear[:6]-=1e-4*previous
    opt=minimize(lambda x:.5*x@Q@x+linear@x,seed.x,jac=lambda x:Q@x+linear,
        method='SLSQP',bounds=Bounds(lo,hi),constraints=[LinearConstraint(eq,b,b),LinearConstraint(U,-np.inf,ub)],
        options={'maxiter':100,'ftol':1e-8})
    x=opt.x
    equilibrium=float(np.max(np.abs(J[:6]@x-h[:6])))
    torque=h[6:]-J[6:]@x
    valid=(opt.success and equilibrium<1e-5 and np.max(U@x-ub)<1e-5)
    if exact:valid=valid and np.linalg.norm((A@x[:6]-desired)[task_rows])<1e-5
    return dict(success=bool(valid),reason='ok' if valid else 'qp_residual',forces=x[:6].reshape(2,3),
        contacts=x[6:].reshape(2,6),torques=torque,achieved=A@x[:6],
        equilibrium_residual=equilibrium,effort_ratio=float(np.max(abs(torque)/effort_limits)),
        wrench_error=float(np.linalg.norm(W@(A@x[:6]-desired))))
