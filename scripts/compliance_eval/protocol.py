"""Versioned, simulator-independent compliance evaluation definitions."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'compliance-v1.1'
SEEDS = [20261104, 20261105, 20261106]
DT = .02

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def cases():
    directions = dict(forward=[1,0,0], backward=[-1,0,0], left=[0,1,0], right=[0,-1,0], up=[0,0,1], down=[0,0,-1])
    out = []
    for side in ('left','right'):
        for name, v in directions.items():
            out.append(dict(name=f'{side}_ee_{name}', group='ee', side=side, direction=v,
                            loads=[[f'{side}_hand_mimic',v]]))
    for name in ('forward','backward','left','right'):
        v=directions[name]
        out.append(dict(name=f'root_{name}',group='locomotion',direction=v,loads=[['pelvis',v]]))
    out.append(dict(name='crouch',group='crouch',direction=[0,0,-1],loads=[['pelvis',[0,0,-1]]]))
    for name, sign in [('left',1),('right',-1)]:
        out.append(dict(name=f'yaw_{name}',group='yaw',sign=sign,loads=[['left_shoulder_roll_link',[-sign,0,0]],['right_shoulder_roll_link',[sign,0,0]]]))
    for name,v,axis,sign in [('forward',[1,0,0],1,1),('backward',[-1,0,0],1,-1),('left',[0,1,0],0,-1),('right',[0,-1,0],0,1)]:
        out.append(dict(name=f'bend_{name}',group='bend',axis=axis,sign=sign,offsets={'torso_link':[0.,0.,.25]},loads=[['torso_link',v],['pelvis',[-x for x in v]]]))
    assert len(out)==23
    return out

def envelope(t):
    if t<=2 or t>=8: return 0.
    if t<3: return .5-.5*np.cos(np.pi*(t-2))
    if t>7: return .5-.5*np.cos(np.pi*(8-t))
    return 1.

def behavior_rows():
    return [dict(c,force_N=f,repeat=r) for c in cases() for f in [5.,10.,20.] for r in range(5)]

def load_arrays(path):
    path=Path(path); meta=json.loads((path/'meta_motion.json').read_text())
    tdmeta=json.loads((path/'_tensordict/meta.json').read_text())
    arrays={}
    for key,m in tdmeta.items():
        if not isinstance(m,dict) or 'dtype' not in m: continue
        arrays[key]=np.memmap(path/'_tensordict'/f'{key}.memmap',mode='r',dtype=m['dtype'].split('.')[-1],shape=tuple(m['shape']))
    return meta,arrays

def prepare(root, output):
    """Select clips only by motion features, never by a policy's performance."""
    from scipy.spatial.transform import Rotation
    root=Path(root); output=Path(output)
    if output.exists(): raise FileExistsError(f'Immutable protocol manifest already exists: {output}')
    sources=[('validation','passive_whole_body_full/val'),('validation','crouch_lowlevel/val'),
             ('train_diagnostic','../artifacts/ceer_segmented_jacobian/20261027-ee-root3-multi10/dataset'),('train_diagnostic','crouch_lowlevel/train')]
    pool=[]; source_hashes={}
    for split,rel in sources:
        path=(root/'dataset'/rel).resolve(); meta,a=load_arrays(path)
        source_hashes[rel]=digest(path/'meta_motion.json')
        jnames=meta['joint_names']; bnames=meta['body_names']
        for ci,(start,end) in enumerate(zip(meta['starts'],meta['ends'])):
            start=int(start); end=min(int(end),start+1000)
            if end-start<251: continue
            q=np.asarray(a['joint_pos'][start:end],float); p=np.asarray(a['root_pos_w'][start:end],float)
            qr=np.asarray(a['root_quat_w'][start:end],float)
            yaw=np.unwrap(Rotation.from_quat(qr[:,[1,2,3,0]]).as_euler('xyz')[:,2])
            def activity(prefix):
                ids=[i for i,n in enumerate(jnames) if n.startswith(prefix) and any(s in n for s in ('shoulder','elbow','wrist'))]
                return float(np.mean(np.ptp(q[:,ids],axis=0)))
            l,r=activity('left_'),activity('right_'); travel=float(np.linalg.norm(p[-1,:2]-p[0,:2])); turn=float(np.ptp(yaw)); z=float(np.ptp(p[:,2]))
            waist=[i for i,n in enumerate(jnames) if n in ('waist_pitch_joint','waist_roll_joint')]
            bend=float(np.max(np.ptp(q[:,waist],axis=0)))
            move=float(np.max(np.ptp(q,axis=0)))
            category=('crouch' if 'crouch_lowlevel' in rel else 'still' if move<.12 and travel<.08 else
                      'locomotion' if travel>.5 else 'yaw' if turn>.5 else 'bend' if bend>.35 else
                      'left_arm' if l>.18 and l>2*r else 'right_arm' if r>.18 and r>2*l else
                      'both_arms' if min(l,r)>.18 else 'mixed')
            pool.append(dict(split=split,dataset=rel,clip=ci,source_motion_id=int(a['motion_id'][start]),source_start_step=int(a['step'][start]),
                             frames=end-start,category=category,features=dict(left=l,right=r,travel=travel,yaw=turn,height_range=z,bend=bend)))
    rng=np.random.default_rng(SEEDS[0]); chosen=[]
    categories=['still','left_arm','right_arm','both_arms','locomotion','crouch','yaw','bend','mixed']
    coverage={}
    for split,count in [('validation',128),('train_diagnostic',32)]:
        buckets={c:[x for x in pool if x['split']==split and x['category']==c] for c in categories}
        coverage[split]={c:len(v) for c,v in buckets.items()}
        for v in buckets.values(): rng.shuffle(v)
        selected=[]
        while len(selected)<count:
            before=len(selected)
            for c in categories:
                if buckets[c] and len(selected)<count:selected.append(buckets[c].pop())
            if len(selected)==before: raise RuntimeError('Not enough unique clips')
        chosen+=selected
    result=dict(version=VERSION,dt=DT,seeds=SEEDS,clips=chosen,source_meta_sha256=source_hashes,available_categories=coverage,
                provenance='validation is a dataset split, not proof of source-recording disjointness; AMASS original recording IDs unavailable',
                behavior=dict(cases=cases(),magnitudes_N=[5,10,20],repeats=5,seconds=12,ramp=[2,3,7,8],
                              coordinate_frame='initial heading, held fixed in world; loads at moving link origins',
                              initial_joint_noise_rad=.003,fall_root_height_m=.30,fall_tilt_deg=75),
                video=dict(width=960,height=540,fps=25,crf=28,maxrate='1200k',bufsize='2400k'))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=prepare(a.root,a.output);print(json.dumps(m['available_categories'],indent=2))
