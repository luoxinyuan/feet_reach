"""Fixed live-root foot targets, quantitative student evaluation, video and web control."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE', '1')
os.environ.setdefault('MEMPATH', str(ROOT / 'dataset'))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', help='Local file, W&B/Forge run URL, or run:entity/project/run-id; defaults to latest pipeline final student')
    p.add_argument('--wandb-file', help='Exact checkpoint path in W&B run Files; defaults to final, then highest numbered checkpoint')
    p.add_argument('--checkpoint-cache', type=Path, default=ROOT / '.cache/wandb-checkpoints')
    p.add_argument('--targets', type=Path, help='JSON: [[x,y,z], ...] in live root frame, metres')
    p.add_argument('--points', type=int, default=20, help='Number of consecutive nearest-neighbor feasible targets')
    p.add_argument('--settle', type=float, default=1.)
    p.add_argument('--reach', type=float, default=2., help='Time allowed to reach the directly commanded target')
    p.add_argument('--hold', type=float, default=1., help='Final measurement window in seconds')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path)
    p.add_argument('--continue-on-instability', action='store_true', help='Record instability but continue all targets without resetting')
    p.add_argument('--video', action='store_true')
    p.add_argument('--web', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--web-seconds', type=float, default=0., help='0 keeps server running until Ctrl+C')
    args = p.parse_args()
    if args.points < 1 or min(args.settle, args.reach, args.hold) <= 0:
        p.error('Counts and durations must be positive')
    if args.checkpoint is None:
        if args.wandb_file:
            p.error('--wandb-file requires --checkpoint with a W&B run')
        if not (ROOT / 'outputs/wall-foot-reach/latest_pipeline.txt').is_file():
            p.error('No local pipeline found; specify --checkpoint with a local file or W&B run URL')
        pipeline = Path((ROOT / 'outputs/wall-foot-reach/latest_pipeline.txt').read_text().strip())
        matches = list((pipeline / 'finetune').rglob('checkpoint_final.pt'))
        if not matches:
            completed = sorted((ROOT/'outputs/wall-foot-reach').glob('*/finetune/**/checkpoint_final.pt'))
            if completed:
                matches = [completed[-1]]
                print(f'Latest pipeline still running; evaluating completed checkpoint: {matches[0]}')
        if len(matches) != 1: p.error('Specify --checkpoint: final finetune checkpoint not unique')
        args.checkpoint = matches[0]
    from scripts.foot_reach.checkpoints import resolve_checkpoint
    try:
        args.checkpoint = resolve_checkpoint(str(args.checkpoint), args.checkpoint_cache, args.wandb_file)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if args.output is None:
        args.output = ROOT / 'artifacts/foot_reach_eval' / time.strftime('%Y%m%d_%H%M%S')
    args.output.mkdir(parents=True, exist_ok=False)
    return args


def select_targets(args):
    import numpy as np
    workspace = np.load(ROOT / 'artifacts/foot_reach/workspace.npz')
    anchor = workspace['points_root'][0]
    if args.targets:
        targets = np.asarray(json.loads(args.targets.read_text()), dtype=float)
    else:
        candidates = np.unique(workspace['points_root'], axis=0)
        candidates = candidates[np.linalg.norm(candidates-anchor, axis=1) > 1e-9]
        if len(candidates) < args.points:
            raise ValueError(f'Only {len(candidates)} distinct non-anchor feasible points; requested {args.points}')
        chosen = []
        current = anchor
        for _ in range(args.points):
            index = int(np.linalg.norm(candidates-current, axis=1).argmin())
            current = candidates[index].copy()
            chosen.append(current)
            candidates = np.delete(candidates, index, axis=0)
        targets = np.asarray(chosen)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all() or not len(targets):
        raise ValueError('Targets must be a finite nonempty Nx3 array')
    return targets, anchor, workspace['points_root'].min(0), workspace['points_root'].max(0)


def write_reports(output, trials, planned_targets=None):
    """Keep failures in denominators; accuracy of completed trials is labelled."""
    points = []
    for point in sorted({r['point'] for r in trials}):
        rows = [r for r in trials if r['point'] == point]
        completed = [r for r in rows if r['completed_hold'] and r['mean_error_m'] is not None]
        points.append(dict(point=point, target_root_xyz_m=rows[0]['target_root_xyz_m'],
            trials=len(rows), successes=sum(r['success'] for r in rows),
            failures=sum(r['failed'] for r in rows),
            mean_error_completed_trials_m=(sum(r['mean_error_m'] for r in completed)/len(completed)
                                           if completed else None)))
    planned_targets = len(trials) if planned_targets is None else planned_targets
    summary = dict(trials=trials, per_point=points,
        planned_targets=planned_targets, attempted_targets=len(trials),
        unattempted_targets=planned_targets-len(trials),
        all_targets_evaluated=len(trials)==planned_targets and all(r['completed_hold'] for r in trials),
        sequence_completed=len(trials)==planned_targets and all(r['success'] for r in trials),
        success_rate=sum(r['success'] for r in trials)/len(trials),
        failure_rate=sum(r['failed'] for r in trials)/len(trials),
        note='Hold frames only. Rates use attempted targets only; unattempted targets are reported separately. Drift is relative to the initial reset. Success means the entire trial stayed stable; accuracy does not determine success. Per-point mean uses completed holds only.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader(); writer.writerows(trials)


def main():
    args = arguments()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import active_adaptation.learning
    from isaaclab.app import AppLauncher
    targets, anchor, lower, upper = select_targets(args)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'vecnorm' not in state or 'cfg' not in state:
        raise ValueError('Checkpoint must include cfg and vecnorm; refusing unnormalised evaluation')
    if state['policy'].get('last_phase') not in ('adapt', 'finetune'):
        raise ValueError('Use a trained student checkpoint from adapt or finetune')
    cfg = OmegaConf.create(OmegaConf.to_container(state['cfg'], resolve=True))
    OmegaConf.set_struct(cfg, False)
    if not cfg.task.command._target_.endswith('.WallFootReachCommand'):
        raise ValueError('Checkpoint is not a wall foot reach policy')
    cfg.checkpoint_path = str(args.checkpoint)
    cfg.vecnorm = 'eval'; cfg.seed = args.seed; cfg.headless = True
    cfg.task.num_envs = 1; cfg.task.max_episode_length = 100000000
    cfg.task.command.student_train = True
    if 'foot_force' in cfg.task.command:
        cfg.task.command.foot_force.enabled = False  # Clean tracking evaluation by default.
    cfg.task.command.disable_motion_finish = True
    cfg.task.command.dataset.fix_ds = 0; cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.termination = {}  # Evaluate failures explicitly, never auto-reset them away.
    cfg.task.viewer.resolution = [960, 720]
    cfg.task.viewer.eye = [2.1, -2.4, 1.5]; cfg.task.viewer.lookat = [0., 0., .75]
    cfg.eval_render = args.video or args.web
    cfg.app = dict(headless=True, enable_cameras=cfg.eval_render)
    cfg.wandb.mode = 'disabled'
    (args.output / 'targets.json').write_text(json.dumps(targets.tolist(), indent=2))
    (args.output / 'config.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        coordinate_frame='live root, full quaternion; left sole centre; metres', seed=args.seed,
        external_force_enabled=False, rng_seed_timing='after checkpoint loading',
        repeats=1, settle_s=args.settle, reach_s=args.reach, hold_s=args.hold,
        protocol='continuous_nearest_targets_v4', continue_on_instability=args.continue_on_instability, default_targets='nearest unvisited feasible point from anchor; full workspace, not validation-only',
        failure='nonfinite state, root displacement >0.20m, gravity z >-0.7, support drift >0.12m or hand drift >0.15m'), indent=2))
    del state
    app = AppLauncher(headless=True, enable_cameras=cfg.eval_render, device='cuda:0').app
    writer = viewer = env = None
    try:
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply, quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        from tensordict import TensorDict
        from PIL import Image, ImageDraw
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); agent.eval()
        # Seed after network construction/loading so checkpoint architecture
        # differences do not change the evaluation random sequence.
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        policy = agent.get_rollout_policy('eval')
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        dt = base.step_dt
        marker = actual_marker = None
        if cfg.eval_render:
            from pxr import UsdGeom, UsdLux, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            # A dedicated render camera avoids headless viewport pose overrides.
            import omni.replicator.core as rep
            camera = UsdGeom.Camera.Define(stage, '/World/FootEvalCamera')
            camera.CreateFocalLengthAttr(24.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateVerticalApertureAttr(27.)
            camera_pose = camera.AddTransformOp()
            base._rgb_annotator.detach()
            base._render_product = rep.create.render_product('/World/FootEvalCamera', (960, 720))
            base._rgb_annotator.attach([base._render_product])
            for path in ('/World/light_0', '/World/light_1'):
                light = UsdLux.DistantLight.Get(stage, path)
                light.GetIntensityAttr().Set(600.)
            def sphere(path, color, radius):
                prim = UsdGeom.Sphere.Define(stage, path)
                prim.CreateRadiusAttr(radius); prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                return UsdGeom.XformCommonAPI(prim)
            marker = sphere('/World/FootTarget', (1., .05, .05), .025)
            actual_marker = sphere('/World/FootActual', (.05, 1., .25), .015)
            base.sim.set_camera_view(cfg.task.viewer.eye, cfg.task.viewer.lookat)
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output / 'evaluation.mp4'), fps=10, codec='libx264', macro_block_size=2)
        if args.web:
            from scripts.foot_reach.web import Viewer
            viewer = Viewer(args.host, args.port)
            print(f'WEB_READY http://{args.host}:{args.port} (forward this port in VS Code)', flush=True)
        is_init = True
        reference = {}
        def reset():
            nonlocal is_init, reference
            env.reset()
            base.episode_length_buf.zero_()
            cmd.set_target_foot_pos_b(anchor)
            reference = dict(root=cmd.asset.data.root_pos_w.clone(),
                support=cmd.asset.data.body_pos_w[:, cmd.right_asset].clone(),
                hand=cmd.asset.data.body_pos_w[:, cmd.hand_asset].clone())
            is_init = True
            if cfg.eval_render:
                centre = reference['root'][0].cpu().numpy() + np.array([0., 0., -.05])
                eye = centre + np.array([1.8, -2.1, .65])
                view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*centre), Gf.Vec3d(0, 0, 1))
                camera_pose.Set(view.GetInverse())

        def measure():
            data = cmd.asset.data
            sole = data.body_pos_w[:, cmd.left_asset] + quat_apply(data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(1, -1))
            actual = quat_apply_inverse(data.root_quat_w, sole - data.root_pos_w)[0]
            target = cmd.target_foot_pos_b()[0]
            root_drift = torch.linalg.vector_norm(data.root_pos_w - reference['root']).item()
            support_drift = torch.linalg.vector_norm(data.body_pos_w[:, cmd.right_asset] - reference['support']).item()
            hand_drift = torch.linalg.vector_norm(data.body_pos_w[:, cmd.hand_asset] - reference['hand']).item()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(data.root_state_w).all())
            reasons = []
            if not finite: reasons.append('nonfinite')
            if root_drift > .20: reasons.append('root_drift')
            if data.projected_gravity_b[0, 2].item() > -.7: reasons.append('tilt')
            if support_drift > .12: reasons.append('support_drift')
            if hand_drift > .15: reasons.append('hand_drift')
            return dict(actual=actual.cpu().numpy(), target=target.cpu().numpy(),
                error=float(torch.linalg.vector_norm(actual-target)), root_drift=root_drift,
                support_drift=support_drift, hand_drift=hand_drift, failure='+'.join(reasons)), sole

        def step(target):
            nonlocal is_init
            cmd.set_target_foot_pos_b(target)
            # Refresh the raw observation AFTER changing the command, then apply
            # the checkpoint normalizer exactly once. Student sees policy only.
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            normalized = normalizer(raw)
            carry = normalized.select('policy')
            carry['is_init'] = torch.full((1,1), is_init, dtype=torch.bool, device=base.device)
            carry = policy(carry)
            base.step(carry)  # no automatic resets, no changing evaluation targets
            is_init = False
            return measure()

        def render_frame(m, sole, label):
            if not cfg.eval_render: return None
            target_world = cmd.target_sole_world()[0].cpu().numpy()
            marker.SetTranslate(Gf.Vec3d(*target_world.astype(float)))
            actual_marker.SetTranslate(Gf.Vec3d(*sole[0].cpu().numpy().astype(float)))
            rgb = base.render('rgb_array')
            if rgb.size == 0: return None
            frame = Image.fromarray(rgb.copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,78), fill=(15,23,42))
            draw.text((12,8), f'{label} | red: target; green: actual sole', fill='white')
            draw.text((12,30), f'Live-root target xyz (m): {np.round(m["target"],3)} | error: {m["error"]*1000:.1f} mm', fill='white')
            draw.text((12,52), f'Failure: {m["failure"] or "none"}', fill='white')
            if writer: writer.append_data(np.array(frame))
            if viewer:
                buffer=io.BytesIO(); frame.save(buffer,format='JPEG',quality=80)
                viewer.publish(buffer.getvalue(), dict(target_xyz_m=np.round(m['target'],4).tolist(),
                    actual_xyz_m=np.round(m['actual'],4).tolist(), error_mm=round(m['error']*1000,2),
                    failure=m['failure'], status=label))
            return frame

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            reset()
            if args.web:
                desired = anchor.copy(); current = anchor.copy(); paused = False
                start = time.monotonic(); iteration = 0
                m, sole = measure()
                while app.is_running() and (args.web_seconds <= 0 or time.monotonic()-start < args.web_seconds):
                    tick = time.monotonic()
                    for command in viewer.take_commands():
                        if command == 'reset':
                            reset(); desired=anchor.copy(); current=anchor.copy(); paused=False
                            m,sole=measure()
                        elif command == 'pause': paused = not paused
                        else:
                            axis='xyz'.index(command[0]); desired[axis]+=.01*(1 if command[1]=='+' else -1)
                            desired=np.clip(desired,lower,upper)
                    if not paused:
                        delta=desired-current; distance=np.linalg.norm(delta)
                        current+=delta*min(1., .1*dt/max(distance,1e-9))
                        m,sole=step(current)
                        if m['failure']: paused=True
                    if iteration % 5 == 0:
                        render_frame(m,sole,'paused' if paused else 'teleop')
                    iteration+=1
                    time.sleep(max(0.,dt-(time.monotonic()-tick)))
            else:
                summaries=[]
                with (args.output/'samples.csv').open('w', newline='') as f:
                    fields=['point','repeat','time_s','phase','target_x','target_y','target_z',
                            'actual_x','actual_y','actual_z','error_m','root_drift_m','support_drift_m','hand_drift_m','failure']
                    csv_writer=csv.DictWriter(f,fieldnames=fields);csv_writer.writeheader()
                    total=0
                    for point,target in enumerate(targets):
                        repeat=0; errors=[]; vectors=[]; failure=''
                        phases=([('settle',args.settle)] if point == 0 else []) + [('reach',args.reach),('hold',args.hold)]
                        for phase,duration in phases:
                            count=max(1,round(duration/dt))
                            for k in range(count):
                                command=anchor if phase=='settle' else target
                                m,sole=step(command)
                                csv_writer.writerow(dict(point=point,repeat=repeat,time_s=total*dt,phase=phase,
                                    **dict(zip(['target_x','target_y','target_z'],m['target'])),
                                    **dict(zip(['actual_x','actual_y','actual_z'],m['actual'])),error_m=m['error'],
                                    root_drift_m=m['root_drift'],support_drift_m=m['support_drift'],hand_drift_m=m['hand_drift'],failure=m['failure']))
                                if total%5==0: render_frame(m,sole,f'point {point}, repeat {repeat}, {phase}')
                                total+=1
                                if phase=='hold' and np.isfinite(m['error']):
                                    errors.append(m['error']); vectors.append(m['actual']-m['target'])
                                if m['failure']:
                                    failure='+'.join(sorted(set(failure.split('+') + m['failure'].split('+')) - {''}))
                                    if not args.continue_on_instability: break
                            if failure and not args.continue_on_instability: break
                        successful=not bool(failure)
                        result=dict(point=point,repeat=repeat,target_root_xyz_m=target.tolist(),
                            failed=bool(failure),failure=failure,completed_hold=len(errors)==max(1,round(args.hold/dt)),success=successful,
                            hold_samples=len(errors),mean_error_m=None,rmse_m=None,p95_error_m=None,max_error_m=None,
                            axis_mae_m=None)
                        if errors:
                            result.update(mean_error_m=float(np.mean(errors)),rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                                p95_error_m=float(np.percentile(errors,95)),max_error_m=float(np.max(errors)),
                                axis_mae_m=np.mean(np.abs(vectors),axis=0).tolist())
                        summaries.append(result)
                        write_reports(args.output, summaries, len(targets))
                        f.flush(); print(json.dumps(result),flush=True)
                        if failure and not args.continue_on_instability:
                            print('Sequence stopped after instability; remaining targets were not attempted.', flush=True)
                            break
                print(f'EVAL_COMPLETE {args.output}',flush=True)
    except KeyboardInterrupt:
        print('Evaluation stopped by user.', flush=True)
    except Exception:
        # Isaac's application shutdown may exit the interpreter before an
        # exception propagates. Report it first and return a nonzero status.
        import traceback
        traceback.print_exc()
        if writer: writer.close()
        if viewer: viewer.close()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        if writer: writer.close()
        if viewer: viewer.close()
        from isaaclab.sim import SimulationContext
        context=SimulationContext.instance()
        if context: context._disable_app_control_on_stop_handle=True
        if env: env.close()
        app.close()


if __name__ == '__main__':
    main()
