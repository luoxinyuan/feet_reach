"""Supervised force-history -> realized next-step joint delta-q residual teacher."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from active_adaptation.learning.ppo.common import make_mlp

HISTORY_OFFSETS=(0,1,2,4,8,16)

def build_model(input_dim=9):
    return nn.Sequential(make_mlp([512,256],norm='before'),nn.Linear(256,29))

def make_force_history(data):
    force=data['force_N'].astype(np.float32)
    ds=data['dataset_id'].astype(np.int64); clip=data['clip_id'].astype(np.int64); step=data['step'].astype(np.int64)
    index={(int(d),int(c),int(s)):i for i,(d,c,s) in enumerate(zip(ds,clip,step))}
    result=np.zeros((len(force),9*len(HISTORY_OFFSETS)),dtype=np.float32)
    for i,(d,c,s) in enumerate(zip(ds,clip,step)):
        for j,offset in enumerate(HISTORY_OFFSETS):
            src=index.get((int(d),int(c),int(s)-offset))
            if src is not None: result[i,j*9:(j+1)*9]=force[src]
    return result

def metrics(y,pred):
    std=np.maximum(y.std(0),1e-5); err=pred-y; moving=np.abs(y)>.01
    cosine=np.sum(y*pred,axis=1)/np.maximum(np.linalg.norm(y,axis=1)*np.linalg.norm(pred,axis=1),1e-9)
    per_joint_r2=1-np.sum(err**2,axis=0)/np.maximum(np.sum((y-y.mean(0))**2,axis=0),1e-12)
    return {'nrmse_per_joint_std':float(np.sqrt(np.mean((err/std)**2))),
        'mae_rad':float(np.mean(np.abs(err))),
        'r2_global':float(r2_score(y,pred,multioutput='variance_weighted')),
        'r2_macro_joint':float(np.mean(per_joint_r2)),
        'moving_joint_sign_accuracy':float(np.mean(np.sign(y[moving])==np.sign(pred[moving]))),
        'moving_joint_count':int(moving.sum()),'delta_q_cosine_median':float(np.median(cosine)),
        'delta_q_cosine_mean':float(np.mean(cosine))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--epochs',type=int,default=500);p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(8)
    data=np.load(a.data,allow_pickle=False); force_history=make_force_history(data); delta=data['delta_q_rad'].astype(np.float32)
    groups=data['dataset_id'].astype(np.int64)*1_000_000+data['clip_id'].astype(np.int64)
    if force_history.shape[1]!=54 or delta.shape[1]!=29 or not np.isfinite(force_history).all() or not np.isfinite(delta).all():
        raise ValueError(f'Invalid supervised data shapes or values: {force_history.shape}, {delta.shape}')
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed); train_idx,val_idx=next(split.split(force_history,delta,groups))
    x_mean=force_history[train_idx].mean(0); x_std=np.maximum(force_history[train_idx].std(0),1e-3); y_scale=np.maximum(delta[train_idx].std(0),1e-4)
    x=torch.from_numpy(((force_history-x_mean)/x_std).astype(np.float32)); y=torch.from_numpy((delta/y_scale).astype(np.float32))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=build_model(54).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    xt=x[train_idx].to(device);yt=y[train_idx].to(device);xv=x[val_idx].to(device);yv=y[val_idx].to(device)
    best=float('inf');best_state=None;best_epoch=0;stale=0
    for epoch in range(a.epochs):
        model.train();order=torch.randperm(len(xt),device=device);last_loss=None
        for idx in order.split(a.batch_size):
            pred=model(xt[idx]);loss=F.smooth_l1_loss(pred,yt[idx],beta=1.)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();last_loss=loss
        model.eval()
        with torch.no_grad(): val_loss=float(F.mse_loss(model(xv),yv))
        if val_loss<best-1e-5:
            best=val_loss;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=30: break
        if (epoch+1)%25==0: print(f'epoch={epoch+1} train_loss={float(last_loss):.6f} val_mse={val_loss:.6f} best={best:.6f}',flush=True)
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad(): pred=(model(x[val_idx].to(device)).cpu().numpy()*y_scale)
    report={'samples':int(len(delta)),'train_samples':int(len(train_idx)),'validation_samples':int(len(val_idx)),
        'train_clips':int(len(np.unique(groups[train_idx]))),'validation_clips':int(len(np.unique(groups[val_idx]))),
        'force_dim':54,'output_dim':29,'history_offsets_frames':list(HISTORY_OFFSETS),
        'points':data['point_names'].tolist(),'joint_names':data['joint_names'].tolist(),
        'architecture':'54 (6 x 9 force history) -> MLP(512,256,LayerNorm-before) -> 29 normalized delta-q; output in radians using saved per-joint scale',
        'best_epoch':best_epoch,'best_validation_mse_normalized':best,
        'validation_metrics':metrics(delta[val_idx],pred),'input_mean_N':x_mean.tolist(),'input_std_N':x_std.tolist(),
        'output_scale_rad':y_scale.tolist(),'data':str(a.data.resolve()),'seed':a.seed}
    torch.save({'model':best_state,'input_mean_N':x_mean,'input_std_N':x_std,'output_scale_rad':y_scale,
        'point_names':data['point_names'],'joint_names':data['joint_names'],'history_offsets_frames':list(HISTORY_OFFSETS),
        'force_input_dim':54,'architecture':report['architecture'],'seed':a.seed},a.output_dir/'residual_teacher.pt')
    (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    m=report['validation_metrics']
    lines=['# 3-point force-history to delta-q residual teacher','',
        f"Grouped held-out validation: {report['train_samples']:,} train and {report['validation_samples']:,} validation samples across disjoint clips.",
        'Input order: six current/past force frames at offsets 0, 1, 2, 4, 8, 16; unavailable pre-clip history is zero.',
        'Output: realized next-step 29-joint delta-q in radians. Same 512/256 LayerNorm trunk and split seed as current-only model.','',
        '| NRMSE ↓ | MAE (rad) ↓ | Global R² ↑ | Macro joint R² ↑ | Moving Δq sign ↑ | Median / mean cosine ↑ |',
        '|---:|---:|---:|---:|---:|---:|',
        f"| {m['nrmse_per_joint_std']:.3f} | {m['mae_rad']:.5f} | {m['r2_global']:.3f} | {m['r2_macro_joint']:.3f} | {m['moving_joint_sign_accuracy']:.3f} | {m['delta_q_cosine_median']:.3f} / {m['delta_q_cosine_mean']:.3f} |",'']
    (a.output_dir/'summary.md').write_text('\n'.join(lines));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
