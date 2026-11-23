"""Matched reference-state evaluation; uses production allocator method bodies."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,argparse,ast,hashlib,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from active_adaptation.utils.segmented_guidance import geometry_jacobians,rot
from active_adaptation.utils.body_guidance import consistent_force,limit_force,slew_force
from active_adaptation.utils.jacobian_guidance import allocate_jacobian_forces,reconstruction
from scripts.compliance_eval.protocol import load_arrays

def production_replay_class():
    # Load these exact simulator-independent methods without importing Isaac.
    path=ROOT/'active_adaptation/envs/mdp/commands/jacobian_compliance.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_candidate','_select','_solve']]
    scope=dict(torch=torch,allocate_jacobian_forces=allocate_jacobian_forces,reconstruction=reconstruction,consistent_force=consistent_force,limit_force=limit_force)
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
    return type('ReferenceReplay',(),{n.name:scope[n.name] for n in methods}),hashlib.sha256(path.read_bytes()).hexdigest()

def replay(q,rp,rq,names,points,geom,seed):
    torch.manual_seed(seed);T=len(q)-1;K=len(points);C,_=production_replay_class();c=C();c.device='cpu';c.num_envs=1;c.rows=torch.arange(1)
    c.slots=torch.zeros(1,3,dtype=torch.long);c.active=torch.ones(1,3,dtype=torch.bool);c.force=torch.zeros(1,3,3);c.desired=c.force.clone();c.remaining=torch.zeros(1);c.releasing=torch.zeros(1,dtype=torch.bool);c.stiffness=torch.ones(1,1,1);c.guidance_enabled=True;c.resample_attempts=2;c.force_penalty=1.;c.smooth_penalty=1.;c.match_net_force=True;c.point_ids=torch.arange(K)
    c.prompt_weights=torch.full((6+len(names),),1/.3);c.prompt_weights[:3]=2.;c.prompt_weights[3:6]=2/.3
    pos,jac=geometry_jacobians(q,rp,rq,names,points,geom)
    yaw=np.arctan2(rot(rq)[:,1,0],rot(rq)[:,0,0]);dyaw=np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)))
    forces=[];planned=[]
    for t in range(T):
        c.asset=SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.tensor(pos[t:t+1],dtype=torch.float32),root_pos_w=torch.tensor(rp[t:t+1],dtype=torch.float32)))
        c.point_jacobians=torch.tensor(jac[t:t+1],dtype=torch.float32)
        c.ref_velocity=torch.tensor((pos[t+1:t+2]-pos[t:t+1])/.02,dtype=torch.float32)
        fd=torch.tensor(60*(rp[t+1:t+2]-rp[t:t+1]),dtype=torch.float32);fd*=40/fd.norm(dim=-1,keepdim=True).clamp_min(40)
        td=torch.tensor([8*dyaw[t]],dtype=torch.float32).clamp(-4,4);c._intent=lambda:(fd,td)
        c.generalized_target=torch.zeros(1,6+len(names));c.generalized_target[:,:3]=fd;c.generalized_target[:,5]=td;c.generalized_target[:,6:]=torch.tensor(80*(q[t+1]-q[t]),dtype=torch.float32)
        if t==0:c._select(c.rows)
        else:
            c.remaining-=.02;c.releasing|=c.remaining<=0
            ready=c.releasing&(c.force.norm(dim=-1).amax(-1)<1e-5);c._select(ready.nonzero().squeeze(-1))
        c._solve()
        # The actor at t sees the force held from the previous physics interval.
        out=torch.zeros(K,3);out[c.slots[0]]=c.force[0];forces.append(out.numpy())
        out=torch.zeros(K,3);out[c.slots[0]]=c.desired[0];planned.append(out.numpy())
        for _ in range(4):c.force.copy_(slew_force(c.force,c.desired,.005))
    return np.asarray(forces),np.asarray(planned)

def spring_replay(point_positions,seed=0):
    """Production 3KP direct spring at nominal reference states.

    The actor-visible force at t is the previous physics interval's applied
    force, matching the production buffer semantics. The nominal body position
    is the current reference position, so spring error is x_ref[t+1]-x_ref[t].
    """
    del seed
    current=torch.zeros(1,3,3,dtype=torch.float32)
    actual=[];planned=[]
    for t in range(len(point_positions)-1):
        raw=80.*torch.tensor(point_positions[t+1:t+2]-point_positions[t:t+1],dtype=torch.float32)
        desired=limit_force(raw,per_point=30.,total=60.)
        actual.append(current[0].clone().numpy())
        planned.append(desired[0].clone().numpy())
        for _ in range(4):
            current=slew_force(current,desired,.005,rate=150.)
    return np.asarray(actual),np.asarray(planned)

def root_targets(rp,rq,horizon=1):
    """Root displacement and rotation vector expressed in the current root frame."""
    T=len(rp)-horizon;R=rot(rq)
    dp_world=rp[horizon:]-rp[:T]
    dp_body=np.einsum('tji,tj->ti',R[:T],dp_world)
    relative=np.einsum('tji,tjk->tik',R[:T],R[horizon:])
    dr=Rotation.from_matrix(relative).as_rotvec()
    return np.concatenate((dp_body,dr),axis=-1).astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--clips',type=int,default=560);p.add_argument('--frames',type=int,default=160);p.add_argument('--seed',type=int,default=20261111);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cache=ROOT/'artifacts/offline_force_baseline/train37_budget60';m=json.loads((cache/'manifest.json').read_text());arrays=[load_arrays(Path(x)) for x in m['dataset_paths']];f37=np.load(cache/'forces.npy',mmap_mode='r')
    original=json.loads((ROOT/'artifacts/ceer_segmented_jacobian/20261027-ee-root3-multi10/manifest.json').read_text());lineage=[r['clip'] for r in original['reports'] for span in r['accepted_spans']]
    points3=[dict(body=b,offset=[0,0,0]) for b in ['pelvis','left_hand_mimic','right_hand_mimic']]
    points9=[dict(body=b,offset=[0,0,0]) for b in ['torso_link']+[s+'_'+b for s in ['left','right'] for b in ['shoulder_roll_link','elbow_link','wrist_roll_link','hand_mimic']]]
    points_spring=[dict(body=b,offset=[0,0,0]) for b in ['pelvis','left_hand_mimic','right_hand_mimic']]
    layouts={'jacobian3':points3,'jacobian9':points9,'spring3':points_spring,'force37':m['points']};key=lambda p:(p['body'],tuple(p['offset']))
    canonical=list(dict.fromkeys(key(p) for ps in layouts.values() for p in ps));maps={k:[canonical.index(key(p)) for p in ps] for k,ps in layouts.items()}
    rng=np.random.default_rng(a.seed);pool=m['clips'];selected=[];seen=set()
    for si,fraction in [(0,.8),(1,.2)]:
        candidates=rng.permutation([c for c in pool if c['source_index']==si])
        count=0
        for c in candidates:
            group=(si,lineage[c['source_clip']] if si==0 else c['source_clip'])
            if group in seen or c['frames']<33:continue
            seen.add(group);selected.append(dict(c,group=str(group)));count+=1
            if count>=round(a.clips*fraction):break
    rng.shuffle(selected)
    # Stratified whole-source-motion split, no adjacent frames cross partitions.
    for si in [0,1]:
        subset=[c for c in selected if c['source_index']==si];n=len(subset)
        for i,c in enumerate(subset):c['split']=0 if i<int(.65*n) else (1 if i<int(.8*n) else 2)
    rows=[];allforces={k:[] for k in [*layouts,'jacobian3_planned','jacobian9_planned','spring3_planned']};records=[];started=time.time();hashsplit={}
    for idx,c in enumerate(selected):
        meta,data=arrays[c['source_index']];st=c['start'];T=min(a.frames,c['frames']-1)
        q=np.asarray(data['joint_pos'][st:st+T+1],dtype=np.float64);rp=np.asarray(data['root_pos_w'][st:st+T+1],dtype=np.float64);rq=np.asarray(data['root_quat_w'][st:st+T+1],dtype=np.float64)
        digest=hashlib.sha256(q.tobytes()).hexdigest()
        if digest in hashsplit:continue
        hashsplit[digest]=c['split'];R=rot(rq[:-1]);gravity=np.einsum('tji,j->ti',R,[0,0,-1]);state=np.c_[q[:-1],gravity,rp[:-1,2]]
        root1=root_targets(rp,rq,1)
        root5=root_targets(rp,rq,5)
        T_eval=len(root5) # keep only transitions with a complete five-frame target
        row=dict(state=state[:T_eval],delta_q=np.diff(q,axis=0)[:T_eval],delta_root_1=root1[:T_eval],
                 delta_root_5=root5,clip=np.full(T_eval,idx),split=np.full(T_eval,c['split']),
                 source=np.full(T_eval,c['source_index']),step=np.arange(T_eval));rows.append(row)
        for name in ['jacobian3','jacobian9']:
            raw,planned=replay(q,rp,rq,meta['joint_names'],layouts[name],m['geometry'],a.seed+idx)
            for suffix,f in [('',raw),('_planned',planned)]:
                padded=np.zeros((T_eval,len(canonical),3),dtype=np.float32);padded[:,maps[name]]=np.einsum('tji,tpj->tpi',R[:T_eval],f[:T_eval]);allforces[name+suffix].append(padded)
        spring_pos,_=geometry_jacobians(q,rp,rq,meta['joint_names'],points_spring,m['geometry'])
        raw,planned=spring_replay(spring_pos)
        for suffix,f in [('',raw),('_planned',planned)]:
            padded=np.zeros((T_eval,len(canonical),3),dtype=np.float32);padded[:,maps['spring3']]=np.einsum('tji,tpj->tpi',R[:T_eval],f[:T_eval]);allforces['spring3'+suffix].append(padded)
        raw=f37[c['force_offset']:c['force_offset']+T_eval];padded=np.zeros((T_eval,len(canonical),3),dtype=np.float32);padded[:,maps['force37']]=np.einsum('tji,tpj->tpi',R[:T_eval],raw);allforces['force37'].append(padded)
        records.append(dict(c,eval_frames=T_eval,eval_id=idx,q_hash=digest))
        if (idx+1)%20==0:print('prepared',idx+1,'/',len(selected),'seconds',round(time.time()-started,1),flush=True)
    float_keys={'state','delta_q','delta_root_1','delta_root_5'}
    out={k:np.concatenate([r[k] for r in rows]).astype(np.float32 if k in float_keys else np.int32) for k in rows[0]}
    out.update({k:np.concatenate(v) for k,v in allforces.items()});np.savez_compressed(a.output/'data.npz',**out)
    manifest=dict(version='force-mapping-v2-root-labels',seed=a.seed,records=records,canonical_points=canonical,joint_names=m['joint_names'],layouts=layouts,production_allocator_sha256=production_replay_class()[1],production_spring_sha256=hashlib.sha256((ROOT/'active_adaptation/envs/mdp/commands/direct_keypoint_spring.py').read_bytes()).hexdigest(),force_cache=str(cache),force_K=m['K'],force_budget=60,point_cap=30,dt=.02,nominal_state=True,state='q + gravity in root + root height, no velocities',targets='joint delta at 1 frame; root-relative SE(3) displacement at 1 and 5 frames',split='source clip grouped 65/15/20; exact q window duplicates dropped',note='Data from training corpus, held out for decoder only. Nominal FK replay uses production allocation/selection; no simulator/policy. force37 is blockwise noncausal optimized cache. Jacobian and spring actual forces have physics/observation lag; planned variants remove it for diagnosis. Spring force law is 80 N/m, 30 N per-point, 60 N total, 150 N/s slew, all three points continuously active.',seconds=time.time()-started)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',len(out['clip']),flush=True)
if __name__=='__main__':main()
