"""Batched kinematics from the same USD joint frames used by PhysX.

Only geometry is read: no motion dataset, reward or target state. Joint FK is
needed to fill CEER's existing wrist interface consistently with the residual.
"""
import torch
from isaaclab.utils.math import quat_mul, quat_conjugate, quat_apply, axis_angle_from_quat

class USDKinematics:
    def __init__(self, usd_path, joint_names, device):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(usd_path)
        edges = []
        def quat(q):
            return torch.tensor([q.GetReal(), *q.GetImaginary()], device=device)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint): continue
            j = UsdPhysics.Joint(prim)
            parent, child = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if not parent or not child: continue
            name = prim.GetName()
            index = joint_names.index(name) if name in joint_names else None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                if index is None: raise ValueError(f'Unknown active joint {name}')
                axis_name = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                axis = torch.zeros(3, device=device); axis['XYZ'.index(axis_name)] = 1.
            elif prim.IsA(UsdPhysics.FixedJoint):
                axis = torch.zeros(3, device=device)
            else: raise ValueError(f'Unsupported joint {name}')
            edges.append((parent[0].name, child[0].name, index,
                torch.tensor(list(j.GetLocalPos0Attr().Get()), device=device), quat(j.GetLocalRot0Attr().Get()),
                torch.tensor(list(j.GetLocalPos1Attr().Get()), device=device), quat(j.GetLocalRot1Attr().Get()), axis))
        # Keep only the ancestors of the two wrists, eliminating unused leg FK.
        needed = {'left_hand_mimic','right_hand_mimic'}
        while True:
            parents = {e[0] for e in edges if e[1] in needed}
            new = needed | parents
            if new == needed: break
            needed = new
        edges = [e for e in edges if e[1] in needed]
        self.edges=[]; available={'pelvis'}
        while edges:
            ready=[e for e in edges if e[0] in available]
            if not ready: raise ValueError('Disconnected USD joint tree')
            for e in ready:
                self.edges.append(e); available.add(e[1]); edges.remove(e)
        assert {'left_hand_mimic','right_hand_mimic'} <= available

    def wrists(self, q):
        n=q.shape[0]
        identity=q.new_zeros(n,4); identity[:,0]=1
        states={'pelvis':(q.new_zeros(n,3),identity)}
        for parent, child, index, p0,r0,p1,r1,axis in self.edges:
            pos,rot=states[parent]
            joint_rot=identity
            if index is not None:
                angle=q[:,index:index+1]/2
                joint_rot=torch.cat((angle.cos(),angle.sin()*axis),-1)
            r0=r0.expand(n,-1); r1=r1.expand(n,-1)
            child_rot=quat_mul(quat_mul(quat_mul(rot,r0),joint_rot),quat_conjugate(r1))
            child_pos=pos+quat_apply(rot,p0.expand(n,-1))-quat_apply(child_rot,p1.expand(n,-1))
            states[child]=(child_pos,child_rot)
        positions=torch.stack([states[s+'_hand_mimic'][0] for s in ('left','right')],1)
        rotations=torch.stack([states[s+'_hand_mimic'][1] for s in ('left','right')],1)
        return positions,rotations

    def interface(self,q):
        pos,rot=self.wrists(q)
        return torch.cat((pos.flatten(1),axis_angle_from_quat(rot).flatten(1)),-1)
