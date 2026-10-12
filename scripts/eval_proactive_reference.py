"""Compare teacher/student on continuous held-out full-body references, without wall."""
import os,sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',type=int,nargs='+',default=[0,4,8])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
    phase=state['policy']['last_phase']
    if phase not in ('train','adapt','finetune'):raise ValueError('Unsupported checkpoint phase')
    if not cfg.task.command._target_.endswith('ProactiveEECommand'):raise ValueError('ProactiveEE checkpoint required')
    cfg.algo.phase=phase
    cfg.checkpoint_path=str(a.checkpoint.resolve());cfg.vecnorm='eval';cfg.seed=42
    cfg.task.num_envs=1;cfg.task.max_episode_length=100000;cfg.task.termination={}
    cfg.task.command.student_train=phase!='train';cfg.task.command.disable_motion_finish=True
    cfg.task.command.no_wall_prob=1.;cfg.task.command.dataset.mem_paths=['ee_reach/val']
    cfg.task.command.dataset.sample_once=True;cfg.task.command.dataset.fix_ds=0;cfg.task.command.dataset.fix_motion_id=a.clips[0]
    cfg.task.viewer.resolution=[960,720];cfg.eval_render=True;cfg.headless=True
    cfg.app=dict(headless=True,enable_cameras=True);cfg.wandb.mode='disabled'
    (a.output/'config.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint.resolve()),phase=phase,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),clips=a.clips,reward=OmegaConf.to_container(cfg.task.reward),reference='held-out continuous full-body clips; no external EE override; no wall; deterministic policy'),indent=2))
    del state
    app=AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app
    writer=None
    try:
        from scripts.utils.helpers import make_env_policy
        from tensordict import TensorDict
        from torchrl.envs.utils import set_exploration_type,ExplorationType
        from pxr import UsdGeom,UsdLux,Gf
        import omni.usd,omni.replicator.core as rep
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        env,agent,vecnorm,_=make_env_policy(cfg);env.eval();agent.eval()
        base=env.base_env;cmd=base.command_manager;asset=cmd.asset
        normalizer=vecnorm.to_observation_norm();normalizer.eval();policy=agent.get_rollout_policy('eval')
        stage=omni.usd.get_context().get_stage();camera=UsdGeom.Camera.Define(stage,'/World/ReferenceEvalCamera')
        camera.CreateFocalLengthAttr(24.);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(27.)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.35,-2.1,1.3),Gf.Vec3d(.15,0,.82),Gf.Vec3d(0,0,1)).GetInverse())
        base._rgb_annotator.detach();base._render_product=rep.create.render_product('/World/ReferenceEvalCamera',(960,720));base._rgb_annotator.attach([base._render_product])
        for path in ('/World/light_0','/World/light_1'):UsdLux.DistantLight.Get(stage,path).GetIntensityAttr().Set(600.)
        sphere=UsdGeom.Sphere.Define(stage,'/World/ReferenceTarget');sphere.CreateRadiusAttr(.025);sphere.CreateDisplayColorAttr([Gf.Vec3f(1,.05,.05)]);marker=UsdGeom.XformCommonAPI(sphere)
        eye=np.array([1.35,-2.1,1.3]);forward=np.array([.15,0,.82])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        assert len(cmd.joint_idx_asset)==len(asset.joint_names)==29
        names=list(asset.joint_names);arm={side:[i for i,n in enumerate(names) if n.startswith(side+'_') and any(s in n for s in ('shoulder','elbow','wrist'))] for side in ('left','right')}
        results=[]
        with torch.inference_mode(),set_exploration_type(ExplorationType.DETERMINISTIC):
            for clip in a.clips:
                cmd.dataset.fix_motion_id=clip;cmd.dataset._populate_buffer_full(target='A');env.set_seed(42);cmd.cloud_counter=0;cmd.set_target_ee_pos_b(None);env.reset()
                wall=cmd.wall.data.root_state_w.clone();wall[:,2]=-3.;cmd.wall.write_root_state_to_sim(wall)
                for _ in range(8):base.sim.render()
                root0=asset.data.root_pos_w.clone();feet0=asset.data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]].clone()
                n=int(cmd.lengths[0])-1;rows=[];qs=[];qrefs=[];vs=[];first_failure=None
                assert int(cmd._motion.motion_id[0,0])==clip, 'Wrong cached reference clip'
                d=a.output/f'clip_{clip}';d.mkdir();writer=imageio.get_writer(str(d/'evaluation.mp4'),fps=25,codec='libx264',macro_block_size=2)
                for step in range(n):
                    raw=TensorDict({},batch_size=[1],device=base.device);base._compute_observation(raw)
                    carry=normalizer(raw)
                    if phase!='train':carry=carry.select('policy','geometry_')
                    carry['is_init']=torch.tensor([[step==0]],device=base.device)
                    base.step(policy(carry))
                    q=asset.data.joint_pos[0].cpu().numpy().copy();qr=np.empty_like(q);qr[cmd.joint_idx_asset.cpu().numpy()]=cmd.target_joint_pos_obs()[0].cpu().numpy()
                    qs.append(q);qrefs.append(qr);vs.append(asset.data.joint_vel[0].cpu().numpy().copy())
                    error=float((asset.data.body_pos_w[:,cmd.hand_asset]-cmd.target_ee_world()).norm())
                    drift=float((asset.data.root_pos_w-root0).norm());feet=float((asset.data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]]-feet0).norm(dim=-1).max())
                    failure=drift>.2 or feet>.12 or float(asset.data.projected_gravity_b[0,2])>-.7
                    if failure and first_failure is None:first_failure=(step+1)*base.step_dt
                    rows.append([(step+1)*base.step_dt,error,drift,feet])
                    if step%2==0:
                        marker.SetTranslate(Gf.Vec3d(*cmd.target_ee_world()[0].cpu().numpy().astype(float)))
                        img=Image.fromarray(base.render('rgb_array').copy());draw=ImageDraw.Draw(img)
                        delta=cmd.target_ee_world()[0].cpu().numpy()-eye;depth=float(delta@forward)
                        if depth>0:
                            x=480+640*float(delta@right)/depth;y=360-640*float(delta@up)/depth
                            draw.ellipse((x-8,y-8,x+8,y+8),outline='red',width=3)
                        draw.rectangle((0,0,960,60),fill=(15,23,42))
                        draw.text((12,8),f'{phase} | validation reference {clip} | t={(step+1)*base.step_dt:.2f}s | red = EE reference',fill='white')
                        draw.text((12,30),f'EE error {error*1000:.1f} mm | feet drift {feet*1000:.1f} mm | reference input: {"full-body joints + privileged state" if phase=="train" else "EE xyz + proprioception"}',fill='white');writer.append_data(np.asarray(img))
                    if not np.isfinite(q).all() or drift>.5:break
                writer.close();writer=None
                qs=np.array(qs);qrefs=np.array(qrefs);vs=np.array(vs);rows=np.array(rows)
                np.savez_compressed(d/'samples.npz',joint_names=names,q=qs,q_ref=qrefs,qd=vs,time=rows[:,0],ee_error=rows[:,1],root_drift=rows[:,2],feet_drift=rows[:,3])
                result=dict(clip=clip,duration_s=float(rows[-1,0]),completed=len(rows)==n,first_stability_failure_s=first_failure,ee_rmse_mm=float(np.sqrt(np.mean(rows[:,1]**2))*1000),ee_mean_mm=float(rows[:,1].mean()*1000),max_feet_drift_mm=float(rows[:,3].max()*1000))
                for side,ids in arm.items():
                    result[side+'_arm']=dict(joint_rmse_rad=float(np.sqrt(np.mean((qs[:,ids]-qrefs[:,ids])**2))),joint_speed_rms_rad_s=float(np.sqrt(np.mean(vs[:,ids]**2))),max_joint_peak_to_peak_rad=float(np.ptp(qs[:,ids],axis=0).max()),reference_max_peak_to_peak_rad=float(np.ptp(qrefs[:,ids],axis=0).max()))
                results.append(result);(d/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
                import matplotlib;matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
                for i in arm['right']:
                    axes[0].plot(rows[:,0],qs[:,i],label=names[i]);axes[0].plot(rows[:,0],qrefs[:,i],ls=':',alpha=.5)
                axes[0].set_ylabel('Right arm q (rad); dotted reference');axes[0].legend(fontsize=7,ncol=2)
                axes[1].plot(rows[:,0],vs[:,arm['right']]);axes[1].set_ylabel('Right arm velocity (rad/s)')
                axes[2].plot(rows[:,0],rows[:,1]*1000);axes[2].set_ylabel('Left EE error (mm)');axes[2].set_xlabel('Time (s)')
                for ax in axes:ax.grid(alpha=.3)
                fig.tight_layout();fig.savefig(d/'tracking.png',dpi=140);plt.close(fig)
        (a.output/'summary.json').write_text(json.dumps(results,indent=2));print('EVAL_COMPLETE',a.output,flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
    finally:
        if writer:writer.close()
        app.close()
if __name__=='__main__':main()
