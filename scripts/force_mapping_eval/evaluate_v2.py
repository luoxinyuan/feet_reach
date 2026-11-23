"""Held-out force-conditioned prediction of joint and root motion."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch

METHODS=['jacobian3','jacobian9','spring3','force37']
TARGET_SCALE=np.asarray([.02]*41,dtype=np.float32) # rad for joints/rotation, m for translation
GROUPS={
    'joint_delta':slice(0,29),
    'root1_translation_m':slice(29,32),
    'root1_rotation_rad':slice(32,35),
    'root5_translation_m':slice(35,38),
    'root5_rotation_rad':slice(38,41),
}

def metrics(y,p):
    e=(y-p)**2
    return dict(nrmse=float(np.sqrt(e.sum()/max((y*y).sum(),1e-15))),
                rmse=float(np.sqrt(e.mean())),mae=float(np.abs(y-p).mean()))

def fit(x,y,split,seed,output,*,model_factory=None):
    torch.manual_seed(seed);device='cuda';tr=np.flatnonzero(split==0);va=np.flatnonzero(split==1);te=np.flatnonzero(split==2)
    if model_factory is None:
        model_factory=lambda: torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,y.shape[1]))
    net=model_factory().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    X=torch.tensor(x,device=device);Y=torch.tensor(y/TARGET_SCALE,device=device);idx=torch.tensor(tr,device=device);v=torch.tensor(va,device=device)
    best=float('inf');stale=0;bestepoch=0;weights=None
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
    torch.save(dict(weights={k:v.cpu() for k,v in weights.items()},best_epoch=bestepoch,input_dim=x.shape[1],output_scale=TARGET_SCALE),output)
    def predict(xx):
        with torch.no_grad():return torch.cat([net(torch.tensor(b,device=device)) for b in np.array_split(xx,max(1,len(xx)//4096))]).cpu().numpy()*TARGET_SCALE
    return predict(x[te]),predict,bestepoch

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);a=p.parse_args();torch.set_num_threads(4)
    d=np.load(a.input/'data.npz');manifest=json.loads((a.input/'manifest.json').read_text());out=a.input/'evaluation';out.mkdir(exist_ok=True);started=time.time()
    state=d['state'];split=d['split'];train=split==0;test=split==2;clip=d['clip'][test];y=np.c_[d['delta_q'],d['delta_root_1'],d['delta_root_5']].astype(np.float32);yt=y[test]
    sm=state[train].mean(0);ss=np.maximum(state[train].std(0),.01);s=(state-sm)/ss
    nforce=d['spring3'].shape[1]*3;zeros=np.zeros((len(s),nforce),np.float32)
    # Use one shared 41-D decoder per method. Group metrics below keep physical units separate.
    reports={};preds={};scales={};methods=['state_only']+METHODS
    for name in methods:
        if name=='state_only':x=np.c_[s,zeros];fm=np.zeros(nforce);fs=np.ones(nforce)
        else:
            f=d[name].reshape(len(s),-1);fm=f[train].mean(0);fs=np.maximum(f[train].std(0),.1);x=np.c_[s,(f-fm)/fs]
        scales[name]=dict(state_mean=sm.tolist(),state_scale=ss.tolist(),force_mean=fm.tolist(),force_scale=fs.tolist())
        reports[name]=[];preds[name]=[]
        for seed in a.seeds:
            pred,predict,epoch=fit(x.astype(np.float32),y,split,seed,out/f'{name}_{seed}.pt');preds[name].append(pred)
            r=dict(seed=seed,best_epoch=epoch,groups={k:metrics(yt[:,sl],pred[:,sl]) for k,sl in GROUPS.items()})
            if name!='state_only':
                rng=np.random.default_rng(seed);xx=x[test].copy();xx[:,len(sm):]=xx[rng.permutation(len(xx)),len(sm):];shuffled=predict(xx);r['shuffled_force']={k:metrics(yt[:,sl],shuffled[:,sl]) for k,sl in GROUPS.items()}
                xx=x[test].copy();active=np.repeat(np.linalg.norm(d[name][test],axis=-1)>1e-5,3,axis=-1)
                xx[:,len(sm):]+=rng.normal(0,.1,(len(xx),nforce)).astype(np.float32)*active/fs
                noisy=predict(xx);r['noise_0_1N_active']={k:metrics(yt[:,sl],noisy[:,sl]) for k,sl in GROUPS.items()}
            reports[name].append(r);print(name,seed,'joint NRMSE',r['groups']['joint_delta']['nrmse'],'root1 translation NRMSE',r['groups']['root1_translation_m']['nrmse'],'epoch',epoch,'elapsed',round(time.time()-started),flush=True)
        (out/'partial.json').write_text(json.dumps(reports,indent=2))
    summary={};rng=np.random.default_rng(20261112);clips=np.unique(clip)
    for name,rs in reports.items():
        summary[name]={}
        for group,sl in GROUPS.items():
            errors=np.array([((pr[:,sl]-yt[:,sl])**2).sum(1) for pr in preds[name]])
            base=np.array([((pr[:,sl]-yt[:,sl])**2).sum(1) for pr in preds['state_only']])
            gain=1-errors.sum(1)/np.maximum(base.sum(1),1e-15)
            # Bootstrap whole held-out clips; average seed losses within each clip.
            ce=np.array([errors[:,clip==c].sum() for c in clips]);cb=np.array([base[:,clip==c].sum() for c in clips]);ci=[]
            for _ in range(1000):
                chosen=rng.integers(0,len(clips),len(clips));ci.append(1-ce[chosen].sum()/max(cb[chosen].sum(),1e-15))
            summary[name][group]=dict(nrmse_mean=float(np.mean([r['groups'][group]['nrmse'] for r in rs])),
                nrmse_seed_std=float(np.std([r['groups'][group]['nrmse'] for r in rs])),
                rmse_mean=float(np.mean([r['groups'][group]['rmse'] for r in rs])),
                mae_mean=float(np.mean([r['groups'][group]['mae'] for r in rs])),
                force_sse_reduction_vs_state=float(gain.mean()),
                force_sse_reduction_clip_bootstrap95=np.quantile(ci,[.025,.975]).tolist())
            if name!='state_only':
                summary[name][group]['shuffled_nrmse']=float(np.mean([r['shuffled_force'][group]['nrmse'] for r in rs]))
                summary[name][group]['noise_0_1N_nrmse']=float(np.mean([r['noise_0_1N_active'][group]['nrmse'] for r in rs]))
    force_stats={}
    for name in METHODS:
        f=d[name][test];norm=np.linalg.norm(f,axis=-1);adj=(clip[1:]==clip[:-1])&(np.diff(d['step'][test])==1)
        slew=np.linalg.norm(np.diff(f,axis=0)[adj],axis=-1)/.02
        force_stats[name]=dict(total_mean_N=float(norm.sum(1).mean()),total_p95_N=float(np.quantile(norm.sum(1),.95)),total_max_N=float(norm.sum(1).max()),point_max_N=float(norm.max()),active_points_mean=float((norm>.05).sum(1).mean()),root_frame_slew_p95_N_s=float(np.quantile(slew,.95)) if len(slew) else 0.)
    np.savez_compressed(out/'predictions.npz',**{k:np.array(v) for k,v in preds.items()},test_clip=clip,target=yt)
    report=dict(summary=summary,per_seed=reports,force_stats=force_stats,groups={k:[sl.start,sl.stop] for k,sl in GROUPS.items()},
        split_counts={str(i):dict(frames=int((split==i).sum()),clips=int(len(np.unique(d['clip'][split==i])))) for i in range(3)},seconds=time.time()-started,
        notes='Single shared 41-D decoder predicts joint Δq, root-local SE(3) at 1 frame and 5 frames. Group scores are computed separately in physical units. Force SSE reduction compares each group against the matched state-only decoder.')
    (out/'normalization.json').write_text(json.dumps(scales));(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
