"""Matched CEER / teacher / student evaluation, physics-rate diagnostics, no video."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys, json, argparse, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
from scipy.spatial.transform import Rotation

CLIPS=['ee_000_1.7.npz','ee_020_1.0.npz','ee_040_1.0.npz','ee_060_1.0.npz','ee_060_1.7.npz']


def longest_run(mask):
    maximum=current=0
    for good in mask:
        current=current+1 if good else 0
        maximum=max(maximum,current)
    return maximum


def summarize(physics, control, duration, wall, failure):
    t=physics[:,0];f=physics[:,1];hits=np.flatnonzero(f>1.)
    first=int(hits[0]) if len(hits) else None
    hold=control[:,0]>duration
    tangent=np.linalg.norm(physics[:,3:5]-physics[:,6:8],axis=1)
    required=physics[:,5]>wall+.005
    good=(f>1.) & (tangent<.03) & required
    # Require 300 ms of uninterrupted physics-rate contact and target accuracy.
    success=not failure and longest_run(good)>=60
    rate=np.maximum(np.diff(f,prepend=0),0)/.005
    impulse=None if first is None or first+20>len(f) else float(f[first:first+20].sum()*.005)
    pre=np.zeros(len(t),dtype=bool) if first is None else (t>=t[first]-.12)&(t<t[first]-.02)
    return dict(success=success,stable=not bool(failure),failure=failure,
        contact=first is not None,contact_time_s=None if first is None else float(t[first]),
        impulse_100ms_Ns=impulse,peak_force_N=float(f.max()),max_rate_N_s=float(rate.max()),
        initial_peak_N=None if first is None else float(f[first:first+20].max()),
        precontact_vx_m_s=float(physics[pre,8].mean()) if pre.any() else None,
        hold_tangent_error_m=float(control[hold,1].mean()) if hold.any() else None,
        hold_xyz_error_m=float(control[hold,2].mean()) if hold.any() else None,
        hold_force_N=float(physics[t>duration,1].mean()) if (t>duration).any() else None,
        max_continuous_good_contact_s=longest_run(good)*.005,
        required_contact_fraction=float((f[required]>1).mean()) if required.any() else None,
        impulse_full_Ns=float(f.sum()*.005),max_root_drift_m=float(control[:,3].max()),
        max_feet_drift_m=float(control[:,4].max()),residual_mean_m=float(np.linalg.norm(control[:,5:8],axis=1).mean()),
        residual_max_m=float(np.linalg.norm(control[:,5:8],axis=1).max()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--controller',choices=['ceer','teacher','student'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
    if args.controller=='teacher':assert state['policy']['last_phase']=='train'
    else:assert state['policy']['last_phase']=='adapt'
    cfg.checkpoint_path=str(args.checkpoint.resolve());cfg.vecnorm='eval';cfg.seed=42
    cfg.task.num_envs=1;cfg.task.max_episode_length=100000;cfg.task.termination={}
    cfg.task.command.student_train=True;cfg.task.command.disable_motion_finish=True
    cfg.task.command.gentle.curriculum_enabled=False
    cfg.task.command.dataset.fix_ds=0;cfg.task.command.dataset.fix_motion_id=0
    cfg.task.eval_contact_forces=True;cfg.eval_render=False;cfg.headless=True
    cfg.app=dict(headless=True,enable_cameras=False);cfg.wandb.mode='disabled'
    metadata=dict(controller=args.controller,checkpoint=str(args.checkpoint.resolve()),
        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),clips=CLIPS,
        walls=[.40,.415],seeds=[42,43,44],hold_s=2.,physics_dt=.005,
        reference='Held-out EE trajectory and both hand orientations; fixed reference root/right hand; common anchor reset',
        teacher='encoder_priv + actor_teacher, deterministic' if args.controller=='teacher' else None,
        impulse='Sum F*0.005 over first 20 physics samples including first F>1 N; matches training integration',
        success='stable trial and >=0.3s continuous 200Hz force>1N, tangent error<3cm, target x>wall+5mm',
        physics_columns=['time','force','hand_x','hand_y','hand_z','target_x','target_y','target_z','vx'],
        control_columns=['time','tangent_error','xyz_error','root_drift','feet_drift','residual_x','residual_y','residual_z',
                         'nominal_x','nominal_y','nominal_z','command_x','command_y','command_z','approach_cost_raw'])
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2));del state
    app=AppLauncher(headless=True,device='cuda:0').app
    try:
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type,ExplorationType
        from tensordict import TensorDict
        env,agent,vecnorm,_=make_env_policy(cfg);env.eval();agent.eval();base=env.base_env;cmd=base.command_manager
        norm=vecnorm.to_observation_norm();norm.eval()
        # eval/deploy in RootStudent always chooses student; teacher must use train path.
        policy=agent.get_rollout_policy('train' if args.controller=='teacher' else 'eval')
        sensor=base.scene['eval_left_hand_wall_force']
        physics=[];clock=0.;world=np.zeros(3)
        def sample(substep):
            nonlocal clock
            clock+=.005
            f=sensor.data.force_matrix_w.reshape(1,-1,3).sum(1)[0]
            pos=cmd.asset.data.body_pos_w[0,cmd.hand_asset]
            vx=cmd.asset.data.body_lin_vel_w[0,cmd.hand_asset,0]
            physics.append([clock,max(0.,-float(f[0])),*pos.cpu().tolist(),*world.tolist(),float(vx)])
        base._post_step_callbacks.append(sample)
        results=[]
        for clipname in CLIPS:
            motion=np.load(ROOT/'artifacts/ee_reach/motions/val'/clipname)
            names=motion['body_names'].tolist();left=names.index('left_hand_mimic');right=names.index('right_hand_mimic')
            rotations=Rotation.from_quat(motion['root_rot']).as_matrix()
            positions=np.einsum('tij,tj->ti',rotations,motion['local_body_pos'][:,left])+motion['root_pos']
            # First maximum excursion is the outward endpoint, before hold/return.
            endpoint=int(np.argmax(np.linalg.norm(positions-positions[0],axis=1)))
            duration=endpoint/50.
            refposes=np.concatenate([motion['local_body_pos'][:,[left,right]].reshape(-1,6),
                Rotation.from_quat(motion['local_body_rot'][:,[left,right]].reshape(-1,4)).as_rotvec().reshape(-1,6)],axis=1)
            for wall in [.40,.415]:
                assert positions[endpoint,0]>wall+.005
                for seed in [42,43,44]:
                    if args.limit and len(results)>=args.limit:break
                    trial=f'{clipname[:-4]}_wall{wall:.3f}_seed{seed}'
                    torch.manual_seed(seed);env.set_seed(seed);cmd.cloud_counter=0
                    cmd.wall_range=(wall,wall);cmd.no_wall_prob=0.
                    cmd.clear_root_and_wrist_6d_reference_override();cmd.set_target_ee_pos_b(None)
                    base.reset();base.episode_length_buf.zero_();cmd.step_schedule(1.)
                    root=cmd.asset.data.root_pos_w.clone();feet=cmd.asset.data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]].clone()
                    init=TensorDict({},[1],device=base.device);base._compute_observation(init)
                    initial=np.concatenate([cmd.asset.data.root_state_w.cpu().numpy().ravel(),cmd.asset.data.joint_pos.cpu().numpy().ravel(),
                                            init['hl_policy'].cpu().numpy().ravel(),init['geometry_'].cpu().numpy().ravel()])
                    physics=[];controls=[];rewards=[];clock=0.;failure=''
                    keys=[(group,k) for group,g in base.reward_groups.items() for k,fn in g.funcs.items() if fn.enabled]
                    with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
                        for step in range(round((duration+2.)/.02)):
                            idx=min(step,endpoint);world=positions[idx].copy()
                            ref=torch.as_tensor(refposes[idx],device=base.device,dtype=torch.float32).reshape(1,12)
                            cmd.set_root_and_wrist_6d_reference_override(ref)
                            cmd.set_target_ee_pos_b(quat_apply_inverse(cmd.asset.data.root_quat_w,
                                torch.as_tensor(world,device=base.device,dtype=torch.float32).reshape(1,3)-cmd.asset.data.root_pos_w))
                            raw=TensorDict({},[1],device=base.device);base._compute_observation(raw)
                            norm(raw);raw['is_init']=torch.tensor([[step==0]],device=base.device)
                            if args.controller=='ceer':raw['action']=torch.zeros(1,3,device=base.device)
                            else:policy(raw)
                            nominal=cmd._get_ee_command_reference() if hasattr(cmd,'_get_ee_command_reference') else base.action_manager._get_ee_command_reference()
                            nominal=nominal[0,:3].clone()
                            residual=torch.tanh(raw['action'][0])*base.action_manager.ee_pos_scale[0,0]
                            before=torch.stack([base.stats[k].clone().flatten()[0] for k in keys])
                            base.step(raw)
                            after=torch.stack([base.stats[k].clone().flatten()[0] for k in keys])
                            rewards.append(((after-before)*base.step_dt).cpu().numpy())
                            actual=cmd.asset.data.body_pos_w[0,cmd.hand_asset].cpu().numpy();error=actual-world
                            drift=float((cmd.asset.data.root_pos_w-root).norm())
                            feet_drift=float((cmd.asset.data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]]-feet).norm(dim=-1).max())
                            controls.append([clock,float(np.linalg.norm(error[1:])),float(np.linalg.norm(error)),drift,feet_drift,
                                *residual.cpu().tolist(),*nominal.cpu().tolist(),*(nominal+residual).cpu().tolist(),float(-cmd.gentle_approach()[0,0])])
                            if not np.isfinite(physics[-1]).all() or not np.isfinite(controls[-1]).all():failure='nonfinite'
                            elif drift>.2:failure='root_drift'
                            elif feet_drift>.12:failure='feet_drift'
                            elif float(cmd.asset.data.projected_gravity_b[0,2])>-.7:failure='tilt'
                            if failure:break
                    arr=np.asarray(physics);ctrl=np.asarray(controls);rew=np.asarray(rewards)
                    result=summarize(arr,ctrl,duration,wall,failure)
                    result.update(trial=trial,clip=clipname,wall=wall,seed=seed,duration=duration,
                        initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                        reward_returns={'/'.join(k):float(v) for k,v in zip(keys,rew.sum(0))})
                    np.savez_compressed(args.output/(trial+'.npz'),physics=arr,control=ctrl,rewards=rew,
                        reward_keys=np.array(['/'.join(k) for k in keys]),initial=initial)
                    results.append(result);(args.output/'summary.json').write_text(json.dumps(results,indent=2))
                    print('TRIAL',len(results),trial,json.dumps(result),flush=True)
        print('SUITE_COMPLETE',len(results),flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle=True
        env.close()
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:app.close()

if __name__=='__main__':main()
