"""Grouped-motion probes: can the force cue predict next-frame reference delta-q?"""
import json, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

p=argparse.ArgumentParser()
p.add_argument('--three',type=Path,required=True)
p.add_argument('--nine',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=20261104)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)

def load(path):
    d=np.load(path,allow_pickle=False)
    return {k:d[k] for k in d.files}

three,nine=load(a.three),load(a.nine)
assert np.array_equal(three['motion_index'],nine['motion_index'])
assert np.array_equal(three['step'],nine['step'])
labels_frame_matched=bool(np.allclose(three['delta_q'],nine['delta_q'],atol=1e-7))
motion_ids=np.unique(three['motion_index'])
rng=np.random.default_rng(a.seed); motion_ids=rng.permutation(motion_ids)
n_test=max(1,round(.2*len(motion_ids))); n_val=max(1,round(.1*(len(motion_ids)-n_test)))
test_ids=set(motion_ids[:n_test].tolist())
val_ids=set(motion_ids[n_test:n_test+n_val].tolist())
train_ids=set(motion_ids[n_test+n_val:].tolist())
print(f'motion split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}',flush=True)

def macro_metrics(y,pred,threshold=.01):
    # y and pred are radians. Macro R2 does not let high-variance joints dominate.
    ss_res=((y-pred)**2).sum(0); ss_tot=((y-y.mean(0))**2).sum(0)
    valid=ss_tot>1e-10
    joint_r2=1-ss_res[valid]/ss_tot[valid]
    moving=np.abs(y)>threshold
    sign=np.sign(y[moving])==np.sign(pred[moving]) if moving.any() else np.array([np.nan])
    return {'nrmse':float(np.sqrt(np.mean(((y-pred)/np.maximum(y.std(0),1e-5))**2))),
            'r2_global':float(1-((y-pred)**2).sum()/max(((y-y.mean())**2).sum(),1e-12)),
            'r2_macro_joint':float(np.mean(joint_r2)),
            'moving_joint_sign_accuracy':float(np.mean(sign)),
            'moving_joint_sample_count':int(moving.sum())}

def rows_mask(x, ids): return np.isin(x, list(ids))

def ridge_probe(xs, y, train_m, test_m, alpha=10.):
    tri=rows_mask(train_m,train_ids); tei=rows_mask(train_m,test_ids)
    scaler=StandardScaler().fit(xs[tri]); Xtr=scaler.transform(xs[tri]); Xte=scaler.transform(xs[tei])
    yscale=np.maximum(y[tri].std(0),1e-4)
    model=Ridge(alpha=alpha).fit(Xtr,y[tri]/yscale)
    pred=model.predict(Xte)*yscale
    return macro_metrics(y[tei],pred)

def mlp_probe(xs,y,motion,seed):
    tri=rows_mask(motion,train_ids); vali=rows_mask(motion,val_ids); tei=rows_mask(motion,test_ids)
    scaler=StandardScaler().fit(xs[tri]); X=scaler.transform(xs).astype(np.float32)
    yscale=np.maximum(y[tri].std(0),1e-4).astype(np.float32)
    yn=(y/yscale).astype(np.float32)
    torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],192),torch.nn.GELU(),
        torch.nn.Linear(192,128),torch.nn.GELU(),torch.nn.Linear(128,y.shape[1])).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-4)
    Xt=torch.from_numpy(X).to(device); yt=torch.from_numpy(yn).to(device)
    tr_idx=torch.from_numpy(np.where(tri)[0]).to(device); va_idx=torch.from_numpy(np.where(vali)[0]).to(device)
    best=float('inf'); best_state=None; patience=0; best_epoch=0
    batch=1024
    for epoch in range(120):
        net.train(); perm=tr_idx[torch.randperm(len(tr_idx),device=device)]
        for ix in perm.split(batch):
            pred=net(Xt[ix]); loss=(pred-yt[ix]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): val=float((net(Xt[va_idx])-yt[va_idx]).square().mean())
        if val < best-1e-5:
            best=val; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_epoch=epoch+1; patience=0
        else:
            patience+=1
            if patience>=12: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): pred=net(Xt[torch.from_numpy(np.where(tei)[0]).to(device)]).cpu().numpy()*yscale
    result=macro_metrics(y[tei],pred); result['best_epoch']=best_epoch
    return result

reports=[]
for name,d in [('3-point',three),('9-point',nine)]:
    state=d['state'].astype(np.float32); force=d['force'].astype(np.float32)
    y=d['delta_q'].astype(np.float32); mids=d['motion_index']
    variants={'state_only':state,'force_only':force,'state_plus_force':np.concatenate([state,force],axis=1)}
    res={'form':name,'motions':int(len(np.unique(mids))),'transitions':int(len(mids)),
         'state_dim':int(state.shape[1]),'force_dim':int(force.shape[1]),
         'point_names':d['point_names'].tolist(),'probes':{}}
    point_force=d['force'].reshape(len(force),len(d['point_names']),3)*30.
    res['force_point_distribution']={}
    for j,pname in enumerate(d['point_names'].tolist()):
        norms=np.linalg.norm(point_force[:,j],axis=1)
        res['force_point_distribution'][pname]={'mean_N':float(norms.mean()),
            'p95_N':float(np.quantile(norms,.95)),'max_N':float(norms.max()),
            'active_fraction':float(np.mean(norms>1e-4))}
    for variant,x in variants.items():
        res['probes'][variant]={'ridge':ridge_probe(x,y,mids, test_ids),
                                'mlp':mlp_probe(x,y,mids,a.seed+len(reports)*10+len(res['probes']))}
    # Group-level RMSE, normalized per-joint by the held-out reference variation.
    names=d['joint_names'].tolist()
    groups={'legs':[n for n in names if any(k in n for k in ['hip','knee','ankle'])],
            'waist':[n for n in names if 'waist' in n],
            'left_arm':[n for n in names if n.startswith('left_') and any(k in n for k in ['shoulder','elbow','wrist'])],
            'right_arm':[n for n in names if n.startswith('right_') and any(k in n for k in ['shoulder','elbow','wrist'])]}
    res['joint_groups']={k:v for k,v in groups.items()}
    reports.append(res)
report={'dataset':'passive_whole_body_full/train','motion_split':{'train':len(train_ids),'validation':len(val_ids),'test':len(test_ids),
    'split_seed':a.seed,'grouped_by':'motion_index'},'test_metrics':reports,
    'same_motion_ids':True,'same_frame_labels':labels_frame_matched,
    'metric_notes':{'nrmse':'per-joint standard-deviation-normalized RMSE; lower is better',
    'r2_macro_joint':'mean held-out R2 over joints; higher is better',
    'moving_joint_sign_accuracy':'sign agreement for |reference delta-q| > 0.01 rad',
    'force_scale':'exact normalized force_gt_ actor input, as stored in the rollout tensordict'},
    'limitations':['The two checkpoints use different low-level policies and training runs; this is a system-level comparison, not a controlled ablation of force-point count alone.',
                   'The actor observation includes the available reference/proprioceptive state; state-plus-force measures incremental predictive information from force beyond that state.',
                   'Probe decodability measures consistency of the force cue as a data encoding, not the learned policy performance.']}
(a.output/'metrics.json').write_text(json.dumps(report,indent=2))
lines=['# Force prompt decodability probe','',f"Grouped split by motion: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} held-out motions.",'',
       'NRMSE is normalized per joint (lower is better); macro R² and moving-joint sign accuracy are higher-is-better. State-plus-force should be compared against state-only to measure added information in the force cue.','',
       '| Form | Input | Probe | NRMSE ↓ | Macro joint R² ↑ | Δq sign accuracy ↑ |','|---|---|---|---:|---:|---:|']
for r in reports:
    for inp in ['state_only','force_only','state_plus_force']:
        for model in ['ridge','mlp']:
            m=r['probes'][inp][model]
            lines.append(f"| {r['form']} | {inp} | {model} | {m['nrmse']:.3f} | {m['r2_macro_joint']:.3f} | {m['moving_joint_sign_accuracy']:.3f} |")
lines+=['','## Caveats','',f"- Same motion IDs: yes; exact frame-label match: {'yes' if labels_frame_matched else 'no'}.",*['- '+x for x in report['limitations']],'']
lines+=['## Force point distribution','','| Form | Point | Mean N | P95 N | Max N | Active frames |','|---|---|---:|---:|---:|---:|']
for r in reports:
    for point,m in r['force_point_distribution'].items():
        lines.append(f"| {r['form']} | {point} | {m['mean_N']:.2f} | {m['p95_N']:.2f} | {m['max_N']:.2f} | {100*m['active_fraction']:.1f}% |")
lines.append('')
(a.output/'summary.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
