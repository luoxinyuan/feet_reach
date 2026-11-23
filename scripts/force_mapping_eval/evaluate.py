"""Independent held-out decoders, conditional ambiguity and force diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def metrics(y,p):
    e=(y-p)**2
    return dict(nrmse=float(np.sqrt(e.sum()/max((y*y).sum(),1e-15))),rmse_rad=float(np.sqrt(e.mean())),mae_rad=float(abs(y-p).mean()))

def fit(x,y,split,seed,output):
    torch.manual_seed(seed);device='cuda';tr=np.flatnonzero(split==0);va=np.flatnonzero(split==1);te=np.flatnonzero(split==2)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,y.shape[1])).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    X=torch.tensor(x,device=device);Y=torch.tensor(y/.02,device=device);idx=torch.tensor(tr,device=device);v=torch.tensor(va,device=device);best=float('inf');stale=0;bestepoch=0
    for epoch in range(100):
        net.train()
        for b in idx[torch.randperm(len(idx),device=device)].split(2048):
            loss=(net(X[b])-Y[b]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():score=float((net(X[v])-Y[v]).square().mean())
        if score<best-1e-6:best=score;weights={k:t.detach().clone() for k,t in net.state_dict().items()};stale=0;bestepoch=epoch+1
        else:stale+=1
        if stale>=12:break
    net.load_state_dict(weights)
    torch.save(dict(weights={k:v.cpu() for k,v in weights.items()},best_epoch=bestepoch,input_dim=x.shape[1]),output)
    def predict(xx):
        with torch.no_grad():return torch.cat([net(torch.tensor(b,device=device)) for b in np.array_split(xx,max(1,len(xx)//4096))]).cpu().numpy()*.02
    return predict(x[te]),predict,bestepoch

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);a=p.parse_args();torch.set_num_threads(4)
    d=np.load(a.input/'data.npz');manifest=json.loads((a.input/'manifest.json').read_text());out=a.input/'evaluation';out.mkdir(exist_ok=True);started=time.time()
    state=d['state'];y=d['delta_q'];split=d['split'];train=split==0;test=split==2;clip=d['clip'][test];yt=y[test];names=manifest['joint_names'];methods=['jacobian3','jacobian9','force37','jacobian3_planned','jacobian9_planned']
    sm=state[train].mean(0);ss=np.maximum(state[train].std(0),.01);s=(state-sm)/ss
    nforce=d['force37'].shape[1]*3;zeros=np.zeros((len(s),nforce),np.float32);reports={};preds={};scales={}
    groups={side+'_arm':[i for i,n in enumerate(names) if n.startswith(side) and any(t in n for t in ['shoulder','elbow','wrist'])] for side in ['left','right']}
    groups['legs']=[i for i,n in enumerate(names) if any(t in n for t in ['hip','knee','ankle'])];groups['waist']=[i for i,n in enumerate(names) if 'waist' in n]
    for name in ['state_only']+methods:
        if name=='state_only':x=np.c_[s,zeros];fm=np.zeros(nforce);fs=np.ones(nforce)
        else:
            f=d[name].reshape(len(s),-1);fm=f[train].mean(0);fs=np.maximum(f[train].std(0),.1);x=np.c_[s,(f-fm)/fs]
        scales[name]=dict(state_mean=sm.tolist(),state_scale=ss.tolist(),force_mean=fm.tolist(),force_scale=fs.tolist())
        reports[name]=[];preds[name]=[]
        for seed in a.seeds:
            pred,predict,epoch=fit(x.astype(np.float32),y,split,seed,out/f'{name}_{seed}.pt');preds[name].append(pred)
            r=dict(seed=seed,best_epoch=epoch,all=metrics(yt,pred),groups={k:metrics(yt[:,ids],pred[:,ids]) for k,ids in groups.items()})
            moving=np.sqrt((yt*yt).mean(1))>=.001;r['moving']=metrics(yt[moving],pred[moving]);r['static']=metrics(yt[~moving],pred[~moving]) if (~moving).any() else None
            r['by_source']={str(i):metrics(yt[d['source'][test]==i],pred[d['source'][test]==i]) for i in [0,1]}
            if name!='state_only':
                rng=np.random.default_rng(seed);xx=x[test].copy();xx[:,len(sm):]=xx[rng.permutation(len(xx)),len(sm):];r['shuffled_force']=metrics(yt,predict(xx))
                xx=x[test].copy();active=np.repeat(np.linalg.norm(d[name][test],axis=-1)>1e-5,3,axis=-1)
                xx[:,len(sm):]+=rng.normal(0,.1,(len(xx),nforce)).astype(np.float32)*active/fs;r['noise_0_1N_active']=metrics(yt,predict(xx))
            reports[name].append(r);print(name,seed,'NRMSE',r['all']['nrmse'],'epoch',epoch,'seconds',round(time.time()-started),flush=True)
        (out/'partial.json').write_text(json.dumps(reports,indent=2))
    summary={};rng=np.random.default_rng(123);clips=np.unique(clip)
    for name,rs in reports.items():
        errors=np.array([((p-yt)**2).sum(1) for p in preds[name]]);base=np.array([((p-yt)**2).sum(1) for p in preds['state_only']]);gain=1-errors.sum(1)/base.sum(1)
        ce=np.array([errors[:,clip==c].sum() for c in clips]);cb=np.array([base[:,clip==c].sum() for c in clips]);ci=[]
        for _ in range(500):
            chosen=rng.integers(0,len(clips),len(clips));ci.append(1-ce[chosen].sum()/cb[chosen].sum())
        summary[name]=dict(nrmse_mean=float(np.mean([r['all']['nrmse'] for r in rs])),nrmse_seed_std=float(np.std([r['all']['nrmse'] for r in rs])),information_gain_mean=float(gain.mean()),information_gain_clip_bootstrap95=np.quantile(ci,[.025,.975]).tolist(),groups={k:float(np.mean([r['groups'][k]['nrmse'] for r in rs])) for k in groups})
        if name!='state_only':summary[name]['noise_nrmse']=float(np.mean([r['noise_0_1N_active']['nrmse'] for r in rs]));summary[name]['shuffled_nrmse']=float(np.mean([r['shuffled_force']['nrmse'] for r in rs]))
    # Same-state neighborhood, different-clip neighbors; no target used to select neighbors.
    xs=s[test];nn=NearestNeighbors(n_neighbors=min(256,len(xs)),n_jobs=4).fit(xs);dist,indices=nn.kneighbors(xs)
    candidates=[];valid=[]
    for i,ii in enumerate(indices):
        ii=ii[clip[ii]!=clip[i]][:32]
        if len(ii)>=8:valid.append(i);candidates.append(ii[:8])
    valid=np.array(valid);cand=np.array(candidates);gaps=((yt[valid,None]-yt[cand])**2).sum(-1);ambiguity={}
    for name in methods[:3]:
        f=d[name][test].reshape(len(xs),-1)/30.;fd=((f[valid,None]-f[cand])**2).sum(-1);best=fd.argmin(1);gap=gaps[np.arange(len(valid)),best]
        ambiguity[name]=dict(state_neighbor_coverage=float(len(valid)/len(xs)),mean_target_gap_state_neighbors=float(gaps.mean()),mean_target_gap_force_nearest=float(gap.mean()),relative_gap_reduction=float(1-gap.mean()/max(gaps.mean(),1e-15)),mean_force_distance=float(np.sqrt(fd.min(1)).mean()),note='Nearest-force among 8 cross-clip state neighbors; descriptive, not proof of injectivity.')
    force_stats={}
    for name in methods:
        f=d[name][test];norm=np.linalg.norm(f,axis=-1);adj=(clip[1:]==clip[:-1])&(np.diff(d['step'][test])==1)
        # Root-frame slew includes changes of coordinates, explicitly labeled.
        slew=np.linalg.norm(np.diff(f,axis=0)[adj],axis=-1)/.02
        force_stats[name]=dict(total_mean_N=float(norm.sum(1).mean()),total_p95_N=float(np.quantile(norm.sum(1),.95)),total_max_N=float(norm.sum(1).max()),point_max_N=float(norm.max()),active_points_mean=float((norm>.05).sum(1).mean()),root_frame_slew_p95_N_s=float(np.quantile(slew,.95)),root_frame_slew_max_N_s=float(slew.max()))
    np.savez_compressed(out/'predictions.npz',**{k:np.array(v) for k,v in preds.items()},test_clip=clip,target=yt)
    report=dict(summary=summary,per_seed=reports,ambiguity=ambiguity,force_stats=force_stats,groups=groups,split_counts={str(i):dict(frames=int((split==i).sum()),clips=int(len(np.unique(d['clip'][split==i])))) for i in [0,1,2]},seconds=time.time()-started)
    (out/'normalization.json').write_text(json.dumps(scales));(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
