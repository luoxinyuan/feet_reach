"""Adapt/student EE accuracy and paired wall/free-space contact evaluation.
All targets are xyz in the live root frame. No automatic resets within trials.
"""
import os,sys,json,csv,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np


def summarize(rows, completed, failure, hold_start):
    reach=[r for r in rows if r['time_s']>=0]
    hold=[r for r in reach if r['time_s']>=hold_start]
    error=np.array([r['error_m'] for r in hold])
    contact=next((r['time_s'] for r in reach if r['normal_force_N']>1.),None)
    already_contact=any(r['normal_force_N']>1. for r in rows if -.1<=r['time_s']<0)
    pre=[r for r in reach if contact is not None and contact-.2<=r['time_s']<contact-.02]
    force=np.array([r['normal_force_N'] for r in reach])
    rate=np.array([r['positive_force_rate_N_s'] for r in reach])
    # A transient in the last window is not a completed reach.
    arrival=None
    for i in range(max(0,len(reach)-39)):
        if all(r['error_m']<=.03 for r in reach[i:i+40]):
            arrival=reach[i]['time_s'];break
    return dict(completed=completed,failure=failure,contact_at_target_start=already_contact,
        contact_time_s=contact,arrival_3cm_for_200ms_s=arrival,
        mean_hold_error_m=float(error.mean()) if len(error) else None,
        rmse_hold_m=float(np.sqrt((error**2).mean())) if len(error) else None,
        p95_hold_error_m=float(np.percentile(error,95)) if len(error) else None,
        within_3cm_hold_fraction=float((error<=.03).mean()) if len(error) else None,
        tracking_success=bool(completed and len(error) and (error<=.03).mean()>=.95),
        peak_force_N=float(force.max()) if len(force) else None,
        max_positive_force_rate_N_s=float(rate.max()) if len(rate) else None,
        force_impulse_Ns=float(force.sum()*.005),
        steady_force_N=float(np.mean([r['normal_force_N'] for r in hold])) if hold else None,
        precontact_mean_vx_m_s=float(np.mean([r['vx_m_s'] for r in pre])) if pre and not already_contact else None,
        precontact_mean_positive_vx_m_s=float(np.mean([max(0.,r['vx_m_s']) for r in pre])) if pre and not already_contact else None,
        precontact_peak_positive_vx_m_s=float(max(max(0.,r['vx_m_s']) for r in pre)) if pre and not already_contact else None,
        precontact_window_samples=len(pre) if not already_contact else 0,
        maximum_root_drift_m=max((r['root_drift_m'] for r in rows),default=0.))


def gradual_target(t, anchor_world, goal_world, wall_x):
    """Fixed world trajectory: 2 s approach, 6 s insertion, 4 s hold."""
    anchor_world=np.asarray(anchor_world);goal_world=np.asarray(goal_world)
    pre=goal_world.copy();pre[0]=wall_x-.06
    if t<=2.:start,end,u=anchor_world,pre,np.clip(t/2.,0.,1.)
    else:start,end,u=pre,goal_world,np.clip((t-2.)/6.,0.,1.)
    blend=10*u**3-15*u**4+6*u**5
    return start+(end-start)*blend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/proactive_ee_eval'/time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--points',type=int,default=7)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--wall-x',type=float,default=.44)
    p.add_argument('--paired-settle',type=float,default=0.,help='0 uses identical reference state/history at target onset; nonzero allows drift during settling')
    p.add_argument('--continue-on-instability',action='store_true',help='Diagnostic only: record threshold failures but continue until severe fall/nonfinite state')
    p.add_argument('--paired-trajectory',action='store_true',help='Matched world trajectory: 2s pre-wall approach, 6s insertion, 4s hold')
    p.add_argument('--video',action='store_true')
    p.add_argument('--masked-control',action='store_true',help='Additional wall-present/pointcloud-masked diagnostic (OOD, not a trained blind baseline)')
    args=p.parse_args()
    if args.points<1 or args.repeats<1 or args.paired_settle<0: p.error('Counts must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['policy']['last_phase'] not in ('adapt','finetune'):raise ValueError('Student checkpoint required')
    cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False)
    if not cfg.task.command._target_.endswith('ProactiveEECommand'):raise ValueError('Wrong task checkpoint')
    cfg.checkpoint_path=str(args.checkpoint.resolve());cfg.vecnorm='eval';cfg.seed=args.seed
    cfg.task.num_envs=1;cfg.task.max_episode_length=10000000;cfg.task.termination={}
    cfg.task.command.student_train=True;cfg.task.command.disable_motion_finish=True
    cfg.task.command.dataset.fix_ds=0;cfg.task.command.dataset.fix_motion_id=0
    cfg.task.eval_contact_forces=True
    cfg.task.viewer.resolution=[960,720]
    cfg.eval_render=args.video;cfg.headless=True;cfg.app=dict(headless=True,enable_cameras=args.video)
    cfg.wandb.mode='disabled'
    manifest=json.loads((ROOT/'artifacts/ee_reach/manifest.json').read_text())
    from scipy.spatial.transform import Rotation
    root=np.array(manifest['q_anchor'][:3]);R=Rotation.from_quat(manifest['q_anchor'][3:7]).as_matrix()
    anchor=R.T@(np.array(manifest['anchor_world'])-root)
    points=np.unique([c['target'] for c in manifest['clips'] if c['split']=='val'],axis=0)
    points=(points-root)@R
    rng=np.random.default_rng(args.seed)
    sampled=points[rng.choice(len(points),min(args.points,len(points)),replace=False)]
    ordered=[];last=anchor
    while len(sampled):
        i=np.linalg.norm(sampled-last,axis=1).argmin();last=sampled[i];ordered.append(last.tolist());sampled=np.delete(sampled,i,0)
    allpoints=np.unique([c['target'] for c in manifest['clips'] if c['split']=='val'],axis=0)
    behind=allpoints[allpoints[:,0]>args.wall_x+.01]
    if not len(behind):raise ValueError('No held-out endpoint behind wall')
    goal_world=behind[np.linalg.norm(behind-np.array(manifest['anchor_world']),axis=1).argmin()];goal=R.T@(goal_world-root)
    if args.paired_trajectory and args.paired_settle!=0:p.error('Trajectory protocol requires zero settling for matched reset state')
    config=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),phase=state['policy']['last_phase'],seed=args.seed,
        accuracy_targets_root=ordered,anchor_root=anchor.tolist(),paired_target_root=goal.tolist(),paired_nominal_target_world=goal_world.tolist(),wall_x=args.wall_x,
        continue_on_instability=args.continue_on_instability,repeats=args.repeats,physics_dt=.005,control_dt=.02,accuracy='seeded random validation endpoints, ordered nearest-neighbor; continuous direct targets, 2s reach + 1s hold; 1s anchor settling',
        paired='nearest held-out target at least 1cm behind wall; wall fixed from reset; optional masked-geometry settling then direct target; 3s reach + 2s hold; default zero settling preserves identical reference-state/history at target onset',paired_settle_s=args.paired_settle,
        contact_threshold_N=1.,precontact_window_s=[-.2,-.02],force_source='left_hand_mimic filtered to reach_wall, 200 Hz normal force',
        geometry='synthetic noisy front-surface point cloud; not rendered depth',masked_control=args.masked_control,
        note='Wall/free-space contrast alone is not proof of compliance. Precontact differences support anticipation; post-contact slowdown can be passive. Masked control is OOD, not a trained blind policy.')
    if args.paired_trajectory:
        config.update(paired='Fixed world quintic waypoints: anchor at 0s, goal yz and wall_x-0.06 at 2s, held-out goal inside wall at 8s, hold until 12s. Converted to live-root xyz at 50Hz.',paired_reach_s=8.,paired_hold_s=4.,wall_thickness_m=.05,video_fps=25)
        if not args.wall_x<goal_world[0]<args.wall_x+.05:raise ValueError('Goal must be inside the wall slab')
        trajectory=np.array([[t,*gradual_target(t,manifest['anchor_world'],goal_world,args.wall_x)] for t in np.arange(0,12.0001,.02)])
        np.savetxt(args.output/'planned_trajectory.csv',trajectory,delimiter=',',header='time_s,x_world,y_world,z_world',comments='')
    (args.output/'config.json').write_text(json.dumps(config,indent=2));del state
    app=AppLauncher(headless=True,enable_cameras=args.video,device='cuda:0').app
    writer=None
    try:
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type,ExplorationType
        from tensordict import TensorDict
        env,agent,vecnorm,_=make_env_policy(cfg);env.eval();agent.eval();base=env.base_env;cmd=base.command_manager
        assert abs(base.physics_dt-.005)<1e-9 and abs(base.step_dt-.02)<1e-9
        normalizer=vecnorm.to_observation_norm();normalizer.eval();policy=agent.get_rollout_policy('eval')
        sensor=base.scene['eval_left_hand_wall_force'];base.sim.set_camera_view([1.4,-2.,1.4],[.2,.0,.8])
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image,ImageDraw
            from pxr import UsdGeom,UsdLux,Gf
            import omni.usd,omni.replicator.core as rep
            stage=omni.usd.get_context().get_stage()
            camera=UsdGeom.Camera.Define(stage,'/World/EEEvalCamera');camera.CreateFocalLengthAttr(24.);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(27.)
            eye=np.array([.10,-2.1,1.22]);centre=np.array([.27,.10,.84]);view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*centre),Gf.Vec3d(0,0,1));camera.AddTransformOp().Set(view.GetInverse())
            base._rgb_annotator.detach();base._render_product=rep.create.render_product('/World/EEEvalCamera',(960,720));base._rgb_annotator.attach([base._render_product])
            for path in ('/World/light_0','/World/light_1'):UsdLux.DistantLight.Get(stage,path).GetIntensityAttr().Set(600.)
            markers=[]
            for name,color,radius in [('Target',(1.,.05,.05),.025),('Actual',(.05,1.,.2),.018)]:
                prim=UsdGeom.Sphere.Define(stage,'/World/EE'+name);prim.CreateRadiusAttr(radius);prim.CreateDisplayColorAttr([Gf.Vec3f(*color)]);markers.append(UsdGeom.XformCommonAPI(prim))
            # A red screen-space ring marks the target even behind the opaque wall.
            forward=(centre-eye)/np.linalg.norm(centre-eye);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            def project(xyz):
                d=xyz-eye;z=float(d@forward)
                return (480+640*(d@right)/z,360-640*(d@up)/z) if z>0 else None
        rows=[];reference={};clock_t=0.;previous_p=None;previous_f=0.;phase='';point=-1;wall_x=args.wall_x;mask=False;fresh=True;present_wall=False
        def physics_sample(substep):
            nonlocal clock_t,previous_p,previous_f
            clock_t+=base.physics_dt
            data=cmd.asset.data;p=data.body_pos_w[0,cmd.hand_asset].detach().cpu().numpy().copy()
            matrix=sensor.data.force_matrix_w
            if matrix is None:raise RuntimeError('Filtered hand-wall force missing')
            force=matrix[0].reshape(-1,3).sum(0).detach().cpu().numpy()
            if not np.isfinite(force).all():raise RuntimeError('Nonfinite contact sensor')
            fn=max(0.,-float(force[0]));vx=0. if previous_p is None else (p[0]-previous_p[0])/base.physics_dt
            actual=quat_apply_inverse(data.root_quat_w,data.body_pos_w[:,cmd.hand_asset]-data.root_pos_w)[0].cpu().numpy()
            target=cmd.target_ee_pos_b()[0].cpu().numpy()
            drift=float((data.root_pos_w-reference['root']).norm());feet=float((data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]]-reference['feet']).norm(dim=-1).max())
            failure=[]
            if not np.isfinite(actual).all() or not torch.isfinite(data.root_state_w).all():failure.append('nonfinite')
            if drift>.20:failure.append('root_drift')
            if feet>.12:failure.append('feet_drift')
            if float(data.projected_gravity_b[0,2])>-.7:failure.append('tilt')
            rows.append(dict(time_s=round(clock_t,6),phase=phase,point=point,error_m=float(np.linalg.norm(actual-target)),
                actual_root_x=float(actual[0]),actual_root_y=float(actual[1]),actual_root_z=float(actual[2]),
                target_root_x=float(target[0]),target_root_y=float(target[1]),target_root_z=float(target[2]),
                hand_world_x=float(p[0]),hand_world_y=float(p[1]),hand_world_z=float(p[2]),
                target_world_x=float(cmd.target_ee_world()[0,0]),target_world_y=float(cmd.target_ee_world()[0,1]),target_world_z=float(cmd.target_ee_world()[0,2]),wall_gap_m=wall_x-float(p[0]),vx_m_s=float(vx),
                normal_force_N=fn,positive_force_rate_N_s=max(0.,(fn-previous_f)/base.physics_dt),
                force_x_N=float(force[0]),force_y_N=float(force[1]),force_z_N=float(force[2]),root_drift_m=drift,feet_drift_m=feet,failure='+'.join(failure)))
            previous_p=p;previous_f=fn
        base._post_step_callbacks.append(physics_sample)
        def reset(present,seed,settle=1.):
            nonlocal rows,reference,clock_t,previous_p,previous_f,fresh,present_wall
            present_wall=present
            fresh=True;env.set_seed(seed);cmd.cloud_counter=0;cmd.no_wall_prob=0. if present else 1.;cmd.wall_range=(wall_x,wall_x)
            cmd.set_target_ee_pos_b(None);env.reset();base.episode_length_buf.zero_();cmd.set_target_ee_pos_b(anchor)
            # Hide the absent collider below the floor as well as outside reach.
            if not present:
                s=cmd.wall.data.root_state_w.clone();s[:,2]=-3.;cmd.wall.write_root_state_to_sim(s)
            reference=dict(root=cmd.asset.data.root_pos_w.clone(),feet=cmd.asset.data.body_pos_w[:,[cmd.left_foot_asset,cmd.right_foot_asset]].clone())
            if args.video:
                for _ in range(8):base.sim.render()
            rows=[];clock_t=-settle;previous_p=cmd.asset.data.body_pos_w[0,cmd.hand_asset].cpu().numpy().copy();previous_f=0.
        def render(label):
            if writer is None:return
            target=cmd.target_ee_world()[0].cpu().numpy().astype(float);actual=cmd.asset.data.body_pos_w[0,cmd.hand_asset].cpu().numpy().astype(float)
            for marker,xyz in zip(markers,[target,actual]):marker.SetTranslate(Gf.Vec3d(*xyz))
            rgb=base.render('rgb_array')
            if not rgb.size:return
            img=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(img)
            for xyz,color in [(target,'red'),(actual,'lime')]:
                pixel=project(xyz)
                if pixel:
                    x,y=pixel;draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
            draw.rectangle((0,0,960,82),fill=(15,23,42));r=rows[-1]
            draw.text((12,7),label+' | target red, actual green (rings remain visible through wall)',fill='white')
            draw.text((12,29),f't={r["time_s"]:.2f}s | error={r["error_m"]*1000:.1f}mm | wall force={r["normal_force_N"]:.1f}N | vx={r["vx_m_s"]:.3f}m/s',fill='white')
            draw.text((12,51),f'target root xyz={np.round(cmd.target_ee_pos_b()[0].cpu().numpy(),3)} | failure={r["failure"] or "none"}',fill='white')
            # Orthographic x-z inset makes the hidden target/wall depth unambiguous.
            box=(620,410,950,705);draw.rectangle(box,fill=(20,28,40),outline='white')
            draw.text((630,420),'Side section (world x-z)',fill='white')
            def inset(xyz):return (635+(xyz[0]-.20)/.36*290,685-(xyz[2]-.55)/.60*230)
            wx,_=inset([wall_x,0,.55]);wx2,_=inset([wall_x+.05,0,.55])
            draw.rectangle((wx,448,wx2,685),fill=(65,80,100) if present_wall else (25,33,45),outline=(100,160,240))
            for xyz,color in [(target,'red'),(actual,'lime')]:
                x,y=inset(xyz);draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
            draw.text((630,445),'red: target / green: hand marker',fill='white')
            draw.text((630,466),'solid wall' if present_wall else 'wall location (absent)',fill='white')
            writer.append_data(np.asarray(img))
        def run_segment(target,seconds,label):
            nonlocal fresh
            if not callable(target):cmd.set_target_ee_pos_b(target)
            for i in range(round(seconds/base.step_dt)):
                if callable(target):
                    world=torch.as_tensor(target(clock_t),dtype=torch.float32,device=base.device).reshape(1,3)
                    cmd.set_target_ee_pos_b(quat_apply_inverse(cmd.asset.data.root_quat_w,world-cmd.asset.data.root_pos_w))
                raw=TensorDict({},batch_size=[1],device=base.device);base._compute_observation(raw)
                if mask:raw['geometry_'].zero_()
                carry=normalizer(raw).select('policy','geometry_');carry['is_init']=torch.tensor([[fresh]],device=base.device)
                base.step(policy(carry));fresh=False
                if i%2==0:render(label)
                if rows[-1]['failure'] and not args.continue_on_instability:return rows[-1]['failure']
                if args.continue_on_instability and (rows[-1]['root_drift_m']>.5 or float(cmd.asset.data.root_pos_w[0,2])<.35 or 'nonfinite' in rows[-1]['failure']):return 'severe_instability:'+rows[-1]['failure']
            return ''
        def save_rows(path):
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        trials=[]
        with torch.inference_mode(),set_exploration_type(ExplorationType.DETERMINISTIC):
            # Free-space accuracy: continuous direct commands, nearby order.
            d=args.output/'accuracy';d.mkdir();mask=False;reset(False,args.seed);phase='settle';point=-1
            if args.video:writer=imageio.get_writer(str(d/'evaluation.mp4'),fps=25,codec='libx264',macro_block_size=2)
            failure=run_segment(anchor,1.,'accuracy | settle');accuracy=[]
            for point,target in enumerate(ordered):
                if failure:
                    accuracy.append(dict(point=point,target_root_xyz=target,attempted=False,failure='previous_failure:'+failure));continue
                phase='reach';start=clock_t;begin=len(rows);failure=run_segment(target,2.,f'accuracy | point {point} reaching')
                phase='hold'
                if not failure:failure=run_segment(target,1.,f'accuracy | point {point} hold')
                part=[dict(r,time_s=r['time_s']-start) for r in rows[begin:]]
                observed_failure=next((r['failure'] for r in part if r['failure']),failure)
                accuracy.append(dict(point=point,target_root_xyz=target,attempted=True,trajectory_finished=not failure,**summarize(part,not observed_failure,observed_failure,2.)))
            save_rows(d/'samples.csv');(d/'summary.json').write_text(json.dumps(accuracy,indent=2))
            if writer:writer.close();writer=None
            conditions=[('no_wall',False,False),('wall',True,False)]
            if args.masked_control:conditions.append(('wall_masked_geometry',True,True))
            for name,present,condition_mask in conditions:
                for repeat in range(args.repeats):
                    d=args.output/name/f'repeat_{repeat}';d.mkdir(parents=True);requested_mask=condition_mask;mask=True;reset(present,args.seed+repeat,args.paired_settle);phase='settle';point=0
                    if args.video:writer=imageio.get_writer(str(d/'evaluation.mp4'),fps=25,codec='libx264',macro_block_size=2)
                    failure=run_segment(anchor,args.paired_settle,name+' | settle')
                    start_state=dict(root_state=cmd.asset.data.root_state_w[0].cpu().tolist(),joint_pos=cmd.asset.data.joint_pos[0].cpu().tolist(),joint_vel=cmd.asset.data.joint_vel[0].cpu().tolist())
                    initial_raw=TensorDict({},batch_size=[1],device=base.device);base._compute_observation(initial_raw)
                    start_state['policy_observation']=initial_raw['policy'][0].cpu().tolist()
                    (d/'start_state.json').write_text(json.dumps(start_state,indent=2))
                    mask=requested_mask
                    phase='reach'
                    reaching=8. if args.paired_trajectory else 3.
                    target_schedule=(lambda t:gradual_target(t,manifest['anchor_world'],goal_world,wall_x)) if args.paired_trajectory else goal
                    if not failure:failure=run_segment(target_schedule,reaching,name+' | slow approach/insertion' if args.paired_trajectory else name+' | reaching')
                    phase='hold'
                    if not failure:failure=run_segment(target_schedule,4. if args.paired_trajectory else 2.,name+' | hold')
                    observed_failure=next((r['failure'] for r in rows if r['failure']),failure)
                    result=dict(condition=name,repeat=repeat,seed=args.seed+repeat,trajectory_finished=not failure,first_stability_failure_s=next((r['time_s'] for r in rows if r['failure']),None),**summarize(rows,not observed_failure,observed_failure,reaching));trials.append(result)
                    save_rows(d/'samples.csv');(d/'summary.json').write_text(json.dumps(result,indent=2))
                    if writer:writer.close();writer=None
                    print('TRIAL',json.dumps(result),flush=True)
        # Paired pre-contact comparison: both windows strictly precede wall contact.
        comparisons=[]
        for repeat in range(args.repeats):
            wall=next(r for r in trials if r['condition']=='wall' and r['repeat']==repeat)
            tc=wall['contact_time_s'];comparison=dict(repeat=repeat,wall_contact_s=tc)
            for name,_,_ in conditions:
                if name=='wall_masked_geometry':continue
                with (args.output/name/f'repeat_{repeat}'/'samples.csv').open() as f:rr=list(csv.DictReader(f))
                selected=[r for r in rr if tc is not None and max(0.,tc-.2)<=float(r['time_s'])<tc-.02]
                comparison[name+'_matched_precontact_vx_m_s']=float(np.mean([float(r['vx_m_s']) for r in selected])) if selected and not wall['contact_at_target_start'] else None
                comparison[name+'_matched_precontact_positive_vx_m_s']=float(np.mean([max(0.,float(r['vx_m_s'])) for r in selected])) if selected and not wall['contact_at_target_start'] else None
            comparisons.append(comparison)
        result=dict(accuracy=accuracy,trials=trials,paired_precontact=comparisons,config=config)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        plot_results(args.output,result)
        print('EVAL_COMPLETE',args.output,flush=True)
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
    finally:
        if writer:writer.close()
        app.close()


def plot_results(output,result):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for condition in sorted({t['condition'] for t in result['trials']}):
        path=output/condition/'repeat_0/samples.csv'
        with path.open() as f:rows=list(csv.DictReader(f))
        t=np.array([float(r['time_s']) for r in rows])
        for ax,key in zip(axes,['wall_gap_m','vx_m_s','normal_force_N']):
            ax.plot(t,[float(r[key]) for r in rows],label=condition)
    for ax,label in zip(axes,['Hand marker gap to wall plane (m)','Hand approach velocity (m/s)','Filtered wall normal force (N)']):
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.axvline(0,color='k',ls=':',lw=1);ax.legend()
    axes[-1].set_xlabel('Time after target command (s)');fig.tight_layout();fig.savefig(output/'paired_curves.png',dpi=150);plt.close(fig)
    with (output/'accuracy/samples.csv').open() as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(11,4));ax.plot([float(r['time_s']) for r in rows],[float(r['error_m'])*1000 for r in rows]);ax.axhline(30,color='r',ls='--');ax.set(xlabel='Time (s)',ylabel='EE error (mm)',title='Free-space continuous EE tracking');ax.grid(alpha=.3);fig.tight_layout();fig.savefig(output/'accuracy_errors.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
