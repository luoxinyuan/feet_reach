"""Export a residual student and its frozen controller for standalone MuJoCo.

Run in the Isaac Lab training environment. Runtime needs only the deploy bundle.
"""
import os,sys,argparse,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import torch
from torch import nn
from omegaconf import OmegaConf
from tensordict import TensorDict
import active_adaptation.learning
from isaaclab.app import AppLauncher

class ResidualActor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(obs_dim,512),nn.LayerNorm(512),nn.Mish(),nn.Linear(512,256),nn.LayerNorm(256),nn.Mish())
        self.mean=nn.Linear(256,29)
    def forward(self,x):return self.mean(self.features(x))

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['sonic','ceer','jacobian','segmented'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--bundle-name',help='Output policy directory name; defaults to backend');args=p.parse_args()
    if args.backend in ('jacobian','segmented') and args.checkpoint is None:p.error('--checkpoint is required for --backend jacobian')
    checkpoint=args.checkpoint or next(Path((ROOT/f'outputs/{args.backend}-residual/latest_full_run.txt').read_text().strip()).rglob('checkpoint_final.pt'))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
    cfg.checkpoint_path=str(checkpoint);cfg.vecnorm='eval';cfg.task.num_envs=1;cfg.wandb.mode='disabled';cfg.eval_render=False
    out=ROOT/f'deploy/mujoco_whole_body/{args.bundle_name or args.backend}';out.mkdir(parents=True,exist_ok=True)
    app=AppLauncher(headless=True,device='cuda:0',kit_args='--/plugins/carb.tasking.plugin/threadCount=16').app
    try:
        from scripts.utils.helpers import make_env_policy
        from torchrl.envs.utils import set_exploration_type,ExplorationType
        env,agent,vecnorm,_=make_env_policy(cfg);env.eval();agent.eval();manager=env.action_manager
        obs_dim=int(vecnorm.loc['policy'].numel())
        actor=ResidualActor(obs_dim).eval()
        source=state['policy']['actor']
        actor.features.load_state_dict({k.removeprefix('module.0.module.1.module.'):v for k,v in source.items() if k.startswith('module.0.module.1.module.')},strict=True)
        actor.mean.load_state_dict({k.removeprefix('module.0.module.2.module.actor_mean.'):v for k,v in source.items() if k.startswith('module.0.module.2.module.actor_mean.')},strict=True)
        torch.jit.script(actor).save(str(out/'student.pt'))
        loc=vecnorm.loc['policy'].cpu();scale=vecnorm.scale['policy'].cpu()
        assert loc.shape==(obs_dim,) and (scale>0).all()
        torch.save(dict(loc=loc,scale=scale),out/'normalization.pt')
        errors=[]
        with torch.inference_mode(),set_exploration_type(ExplorationType.MODE):
            native=agent.get_rollout_policy('deploy');assert list(native.in_keys)==['policy']
            for _ in range(4):
                x=torch.randn(16,obs_dim);y=native(TensorDict({'policy':x.to(env.device)},[16]))['action'].cpu()
                errors.append((actor(x)-y).abs().max().item())
        assert max(errors)<1e-4,errors
        ids=manager.joint_ids;asset=manager.asset
        enabled=[k for k,v in cfg.task.observation.policy.items() if v.get('enabled',True)]
        assert enabled==['root_ang_vel_history','projected_gravity_history','joint_pos_history','prev_actions','upper_body_forces_b','sonic_joint_velocity','sonic_reference']
        def values(x):return x.detach().cpu().tolist()
        meta=dict(backend=args.backend,checkpoint_name=checkpoint.parent.parent.name+'/'+checkpoint.name,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),joint_names=manager.joint_names,
            default_pos=values(asset.data.default_joint_pos[0,ids]),kp=values(asset.data.joint_stiffness[0,ids]),
            kd=values(asset.data.joint_damping[0,ids]),armature=values(asset.data.joint_armature[0,ids]),
            effort_limit=values(asset.data.joint_effort_limits[0,ids]),
            joint_lower=values(manager.reference.lower[0]),joint_upper=values(manager.reference.upper[0]),
            action_scale=values(manager.action_scaling),policy_dt=.02,physics_dt=.005,
            root_height=float(asset.data.default_root_state[0,2]),obs_dim=obs_dim,force_slice=[238,obs_dim-116],
            force_limit=30.,force_slew=150.,force_budget=60.,reference_speed=manager.reference.speed,
            max_reference_error=manager.reference.max_error,observation_terms=enabled,
            native_student_max_error=max(errors),self_collisions=bool(asset.cfg.spawn.articulation_props.enabled_self_collisions))
        if cfg.task.command.get('force_points'):
            meta['force_points']=OmegaConf.to_container(cfg.task.command.force_points,resolve=True)
        if args.backend in ('ceer','jacobian','segmented'):
            from active_adaptation.utils.ceer_residual import TARGET_JOINT_NAMES
            low=manager.low_policy.cpu().eval()
            example=(torch.zeros(1,257),torch.zeros(1,5,29))
            exported=torch.jit.trace(low,example,check_trace=True)
            exported.save(str(out/'low_level.pt'))
            with torch.inference_mode():
                x=torch.randn(32,257);q=torch.randn(32,5,29)
                err=(low(x,q)-exported(x,q)).abs().max().item()
            assert err<1e-6
            meta.update(low_level_sha256=low.checkpoint_sha256,low_level_export_max_error=err,
                target_order=[meta['joint_names'].index(n) for n in TARGET_JOINT_NAMES],
                ceer_force_limit=manager.ceer_force_limit,low_action_config=OmegaConf.to_container(low.action_cfg,resolve=True),
                fk_edges=[[parent,child,index,*[values(t) for t in (p0,r0,p1,r1,axis)]] for parent,child,index,p0,r0,p1,r1,axis in manager.kinematics.edges])
        else:
            for name in ['model_encoder.onnx','model_decoder.onnx']:
                shutil.copy2(ROOT/'.cache/sonic'/name,out/name)
            shutil.copy2(ROOT/'.cache/sonic/source/LICENSE',out/'SONIC_LICENSE.txt')
            meta['onnx_resources']={name:dict(url=f'https://huggingface.co/nvidia/GEAR-SONIC/resolve/bfd764a/{name}',sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('model_encoder.onnx','model_decoder.onnx')}
            meta['sonic_manifest']=json.loads((ROOT/'.cache/sonic/manifest.json').read_text())
        if args.backend in ('ceer','jacobian','segmented'):
            sys.path.insert(0,str(ROOT/'scripts'))
            from mujoco_residual import wrist_interface
            import numpy as np
            q=torch.tensor(meta['default_pos'],device=env.device)[None]+.3*torch.randn(32,29,device=env.device)
            native_fk=manager.kinematics.interface(q).cpu().numpy()
            portable_fk=np.stack([wrist_interface(row,meta['fk_edges']) for row in q.cpu().numpy()])
            meta['native_fk_max_error']=float(np.max(np.abs(native_fk-portable_fk)))
            assert meta['native_fk_max_error']<1e-5
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        (out/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json' and p.suffix!='.onnx'},indent=2)+'\n')
        print('RESIDUAL_EXPORT_COMPLETE',args.backend,max(errors),flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:app.close()
if __name__=='__main__':main()
