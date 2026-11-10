"""Vectorized first-episode tracking and paired fixed-force trials (no auto resets)."""
import os
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('MEMPATH',str(ROOT/'dataset'))
import numpy as np
from scripts.compliance_eval.protocol import behavior_rows,envelope,digest
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--suite',choices=['tracking','behavior'],required=True);p.add_argument('--zero-residual',action='store_true');p.add_argument('--zero',action='store_true');p.add_argument('--seed-index',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--steps',type=int);p.add_argument('--video-output',type=Path);p.add_argument('--video-case');p.add_argument('--video-clip',type=int);p.add_argument('--case');p.add_argument('--force-N',type=float);p.add_argument('--body-relative-bend',action='store_true',help='Recompute bend_forward forces along torso local +X every physics substep');p.add_argument('--guidance-cache',type=Path);p.add_argument('--guidance-cache-map',type=Path,help='JSON mapping each tracking dataset path to an ordered list of compatible cache directories')
p.add_argument('--add-hands-ee-yaw-case',action='store_true',help='Add an equal-and-opposite forward-force couple at both hand EEs for focused yaw diagnostics')
p.add_argument('--offline-prompt-cache-map',type=Path,help='Source-aware offline force-prompt map for OfflineForcePromptCommand');p.add_argument('--ceer-direct-joint-target',action='store_true',help='Bypass the residual actor and feed the current/future full-body reference joint targets directly to the frozen CEER low-level actor')
p.add_argument('--prompt-only-behavior',action='store_true',help='Provide offline force prompt tokens while disabling physical force dispatch')
p.add_argument('--preserve-training-sim',action='store_true',help='Keep checkpoint training rewards, terminations, randomization, and episode length for a fixed-motion diagnostic rollout')
p.add_argument('--supervised-deltaq-checkpoint',type=Path,help='Use standalone force->physical delta-q model instead of the checkpoint PPO actor')
p.add_argument('--supervised-q-input-source',choices=['measured','reference'],default='measured',help='Use measured robot joint state (deployment default) or SONIC persistent reference as the supervised model q input')
p.add_argument('--estimator-audit',action='store_true',help='Measure student direct joint-target estimates and force-token consistency during tracking')
p.add_argument('--policy-branch',choices=['deploy','teacher'],default='deploy',help='Evaluate deployed student branch or, for RootStudentPPO train checkpoints, the privileged teacher actor')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();manifest=json.loads(a.manifest.read_text());video=a.video_case is not None or a.video_clip is not None
if a.suite=='tracking':
    rows=manifest['clips']; rows=[rows[a.video_clip]] if video else rows
else:
    rows=behavior_rows()
    if a.add_hands_ee_yaw_case:
        rows += [dict(name='hands_ee_yaw_couple',group='yaw',sign=-1.,direction=[1.,0.,0.],
                      loads=[['left_hand_mimic',[1.,0.,0.]],['right_hand_mimic',[-1.,0.,0.]]],
                      force_N=f,repeat=r)
                 for f in [5.,10.,20.] for r in range(5)]
    if a.case:
        rows=[dict(r) for r in rows if r['name']==a.case and r['force_N']==10]
        if not rows: raise ValueError(f'Unknown behavior case: {a.case}')
        if a.force_N is not None:
            for row in rows: row['force_N']=a.force_N
    elif a.force_N is not None and not video:
        raise ValueError('--force-N requires --case for focused behavior evaluation')
    if video:
        target_force=a.force_N if a.force_N is not None else 10
        available={r['force_N'] for r in rows if r['name']==a.video_case}
        lookup_force=target_force if target_force in available else 10
        rows=[dict(next(r for r in rows if r['name']==a.video_case and r['force_N']==lookup_force and r['repeat']==0))]
        rows[0]['force_N']=target_force
    if a.body_relative_bend and any(r['name']!='bend_forward' for r in rows):
        raise ValueError('--body-relative-bend can only be used with bend_forward')
    if a.body_relative_bend:
        for row in rows:
            row['force_frame']='torso_local_x_recomputed_each_physics_substep'
            row['name']='bend_forward_body_relative'
if a.limit: rows=rows[:a.limit]
n=len(rows);seed=manifest['seeds'][a.seed_index]
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(state['cfg'],resolve=True));OmegaConf.set_struct(cfg,False);OmegaConf.set_readonly(cfg,False)
cfg.checkpoint_path=str(a.checkpoint.resolve());cfg.vecnorm='eval';cfg.seed=seed;cfg.task.num_envs=n;cfg.wandb.mode='disabled';cfg.eval_render=video;cfg.headless=True
if video:cfg.task.viewer.resolution=[960,540]
if not a.preserve_training_sim:
    cfg.task.reward={};cfg.task.termination={};cfg.task.max_episode_length=2000;cfg.task.observe_on_reset=True;cfg.task.viewer.resolution=[960,540]
    # Stable, common simulator evaluation, keeping checkpoint-specific actor and force generator.
    cfg.task.randomization={}
paths=list(dict.fromkeys(r['dataset'] for r in rows)) if a.suite=='tracking' else list(cfg.task.command.dataset.mem_paths)
cfg.task.command.dataset.mem_paths=paths;cfg.task.command.dataset.path_weights=[1.]*len(paths)
if a.suite=='tracking' and hasattr(cfg.task.command.dataset,'motion_allowlists'):
    # Tracking replays the manifest's validation datasets, not the training
    # datasets named by the checkpoint's allowlists. Keep list lengths aligned
    # and leave those validation datasets unrestricted.
    cfg.task.command.dataset.motion_allowlists=[None]*len(paths)
cfg.task.command.dataset.resample_on_reset=False;cfg.task.command.dataset.sample_once=True;cfg.task.command.dataset.fix_motion_id=None;cfg.task.command.dataset.fix_ds=None
cfg.task.command.guidance_enabled=a.suite=='tracking'
# GT-reference-delta-q prompts are generated from the active reference clip.
# Tracking evaluation intentionally swaps the training corpus for fixed
# validation clips, while the command still validates the original corpus
# metadata against its cache manifest.
if a.suite=='tracking' and str(cfg.task.command.get('_target_','')).endswith('.GTDeltaQCommand'):
    cfg.task.command.allow_dataset_override=True
if a.suite=='behavior':
    for group in cfg.task.observation.values():
        for params in group.values():
            if isinstance(params,dict) or OmegaConf.is_dict(params):
                if 'noise_std' in params:params.noise_std=0.

offline_prompt_map=json.loads(a.offline_prompt_cache_map.read_text()) if a.offline_prompt_cache_map else None
if offline_prompt_map and a.suite!='tracking':
    raise ValueError('--offline-prompt-cache-map is only valid for tracking')
command_target=str(cfg.task.command.get('_target_',cfg.task.command.get('class_name','')))
if a.prompt_only_behavior and (a.suite!='behavior' or 'OfflineForcePrompt' not in command_target):
    raise ValueError('--prompt-only-behavior requires behavior suite and an OfflineForcePromptCommand checkpoint')
if offline_prompt_map and 'guidance_cache' in cfg.task.command:
    cfg.task.command.guidance_cache=None
if 'guidance_cache' in cfg.task.command:
    if a.suite=='tracking' and not a.guidance_cache and not offline_prompt_map: raise ValueError('Offline policies require a matching --guidance-cache; never substitute an unrelated cache')
    if a.guidance_cache:cfg.task.command.guidance_cache=str(a.guidance_cache.resolve())
if a.guidance_cache_map:
    guidance_cache_map=json.loads(a.guidance_cache_map.read_text())
    if a.suite!='tracking': raise ValueError('--guidance-cache-map is only valid for tracking')
else:
    guidance_cache_map={}
phase=cfg.algo.get('phase');saved_config=OmegaConf.to_container(cfg,resolve=True);del state
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=video,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/omni.tbb.globalcontrol/maxThreadCount=8').app
try:
    from scripts.utils.helpers import make_env_policy
    if offline_prompt_map:
        from active_adaptation.envs.mdp.commands.offline_force_prompt import OfflineForcePromptCommand
        from active_adaptation.envs.mdp.commands.sonic_compliance import SonicComplianceCommand
        original_offline_init=OfflineForcePromptCommand.__init__
        original_offline_solve=OfflineForcePromptCommand._solve
        prompt_source_manifest=json.loads((ROOT/'artifacts/offline_force_baseline/train37_budget60/manifest.json').read_text())
        def eval_offline_prompt_init(self,env,*args,guidance_cache=None,physical_guidance=False,**kwargs):
            # Use the same command/observation contract, but inject reference
            # prompts from a separately verified validation cache below.
            SonicComplianceCommand.__init__(self,env,*args,**kwargs)
            from active_adaptation.utils.sonic import JOINT_NAMES
            self.reference_target_ids=[self.dataset.joint_names.index(n) for n in JOINT_NAMES]
            self.cache_manifest=prompt_source_manifest
            self.point_names=[point['name'] for point in prompt_source_manifest['points']]
            self.point_force_buffer=torch.zeros(self.num_envs,len(self.point_names),3,device=self.device)
            self.physical_guidance=bool(physical_guidance)
            self.prompt_body_ids=torch.tensor([self.asset.body_names.index(point['body']) for point in prompt_source_manifest['points']],device=self.device)
            self.prompt_offsets=torch.tensor([point['offset'] for point in prompt_source_manifest['points']],device=self.device,dtype=torch.float32)
            self.force_apply_world=self.physical_guidance
        OfflineForcePromptCommand.__init__=eval_offline_prompt_init
        # Avoid indexing the training cache during the environment's initial
        # reset/update, before manifest rows are loaded into the command.
        OfflineForcePromptCommand._solve=lambda self: None
    # Older training checkpoints can contain normalizer statistics for
    # auxiliary observation groups whose shapes have since changed. The
    # deployed actor only needs its own input normalizer, so retain all
    # shape-compatible stats and require the actor's policy stats to match.
    from tensordict import TensorDict
    from torchrl.envs.transforms import VecNorm

    def load_compatible_vecnorm_state(self, state):
        if state is None:
            raise KeyError('Missing VecNorm state in checkpoint')
        saved = TensorDict(state).flatten_keys(self.SEP)
        current = self._td.flatten_keys(self.SEP) if self._td is not None else TensorDict({}, batch_size=[])
        compatible = {}
        for key in saved.keys():
            value = saved.get(key)
            existing = current.get(key, None)
            if existing is None or existing.shape != value.shape:
                if key in ('policy_sum', 'policy_ssq', 'policy_count'):
                    raise RuntimeError(f'Checkpoint policy normalizer mismatch for {key}: saved={tuple(value.shape)}, env={None if existing is None else tuple(existing.shape)}')
                continue
            compatible[key] = value
        current.update_(TensorDict(compatible, batch_size=[]))
        self._td = current.unflatten_keys(self.SEP)

    VecNorm.set_extra_state = load_compatible_vecnorm_state
    # The critic is never queried during rollout. Some older checkpoints have
    # critic-only privileged inputs that changed shape while the deployed
    # actor input remained identical. Preserve the actor state exactly and
    # skip only incompatible critic tensors in this throwaway process.
    from active_adaptation.learning.hierarchical.dense_force_ppo import DenseForcePPO
    from torch.nn.parallel import DistributedDataParallel
    original_dense_load = DenseForcePPO.load_state_dict

    def load_dense_for_eval(self, state_dict, strict=True):
        state_dict = dict(state_dict)

        def unwrap_ddp(module):
            return module.module if isinstance(module, DistributedDataParallel) else module

        # TensorDictModule also has a `.module` child; only unwrap actual DDP
        # wrappers. Validate the complete actor state before allowing the
        # non-strict load needed for critic-only shape differences.
        actor = unwrap_ddp(self.actor)
        saved_actor = state_dict.get('actor', {})
        actor_state = actor.state_dict()
        actor_mismatch = [
            key for key, value in actor_state.items()
            if key not in saved_actor or saved_actor[key].shape != value.shape
        ]
        actor_extra = sorted(set(saved_actor) - set(actor_state))
        if actor_mismatch or actor_extra:
            raise RuntimeError(f'Actor checkpoint is incompatible: missing/mismatched={actor_mismatch}, extra={actor_extra}')
        critic = unwrap_ddp(self.critic)
        saved_critic = state_dict.get('critic', {})
        compatible_critic = {
            key: value for key, value in saved_critic.items()
            if key in critic.state_dict() and critic.state_dict()[key].shape == value.shape
        }
        skipped = sorted(set(saved_critic) - set(compatible_critic))
        if skipped:
            print(f'[eval] skip incompatible critic-only tensors: {skipped}', flush=True)
        state_dict['critic'] = compatible_critic
        # Non-strict loading now applies only to compatible critic keys: the
        # actor key set and tensor shapes were checked above and remain intact.
        return original_dense_load(self, state_dict, strict=False)

    DenseForcePPO.load_state_dict = load_dense_for_eval
    if guidance_cache_map:
        # Environment construction performs one command update before the
        # source-specific solver below can be attached to its instance.
        from active_adaptation.envs.mdp.commands.segmented_jacobian_compliance import SegmentedJacobianComplianceCommand
        original_segmented_solve=SegmentedJacobianComplianceCommand._solve
        def bootstrap_solve(self):
            try:
                original_segmented_solve(self)
            except (RuntimeError,IndexError):
                self.active.zero_();self.slots.zero_();self.desired.zero_();self.point_desired.zero_()
        SegmentedJacobianComplianceCommand._solve=bootstrap_solve
    from torchrl.envs.utils import set_exploration_type,ExplorationType
    from active_adaptation.utils.math import quat_apply_inverse,quat_apply,quat_mul,quat_conjugate
    from active_adaptation.utils.simple_locomotion_guidance import yaw_from_quat
    torch.manual_seed(seed);np.random.seed(seed)
    env,agent,vecnorm,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval()
    if a.estimator_audit:
        target_loc=vecnorm.loc['joint_target'].to(env.device)
        target_scale=vecnorm.scale['joint_target'].to(env.device)
    if offline_prompt_map:
        OfflineForcePromptCommand.__init__=original_offline_init
        OfflineForcePromptCommand._solve=original_offline_solve
    if guidance_cache_map:SegmentedJacobianComplianceCommand._solve=original_segmented_solve
    cmd=env.command_manager;robot=cmd.asset;manager=env.action_manager;device=env.device
    if a.ceer_direct_joint_target:
        from active_adaptation.envs.mdp.ceer_residual import CEERResidualAction
        from active_adaptation.utils.math import quat_apply,quat_apply_inverse,yaw_quat
        import types
        if not isinstance(manager,CEERResidualAction):
            raise TypeError('--ceer-direct-joint-target requires CEERResidualAction')
        motion_joint_ids=[cmd.dataset.joint_names.index(name) for name in manager.joint_names]
        original_low_inputs=manager.low_inputs
        def direct_target_low_inputs(self,joint_history):
            obs,_=original_low_inputs(joint_history)
            # The frozen CEER was trained with a six-dimensional target
            # command: target root height, planar velocity, target heading,
            # and force limit. This command class does not expose command(),
            # so derive it from the active reference motion as in
            # LowLevelForceRoot3Command.command().
            current_yaw=yaw_quat(robot.data.root_quat_w)
            target_yaw=yaw_quat(cmd._motion.root_quat_w[:,0])
            target_velocity_b=quat_apply_inverse(current_yaw,cmd._motion.root_lin_vel_w[:,0])[:,:2]
            forward=torch.zeros(n,3,device=device);forward[:,0]=1.
            target_heading_w=quat_apply(target_yaw,forward)
            target_heading_b=quat_apply_inverse(current_yaw,target_heading_w)[:,:2]
            root_height=cmd._motion.root_pos_w[:,0,2:3]
            force_limit=torch.full_like(root_height,30.)
            obs[:,1:7]=torch.cat((root_height,target_velocity_b,target_heading_b,force_limit),dim=-1)
            # Preserve the low-level's trained five-frame horizon and pass the
            # fixed motion command's [0,2,4,8,16]-step whole-body targets.
            target=cmd._motion.joint_pos[:,:,motion_joint_ids]
            return obs,target
        manager.low_inputs=types.MethodType(direct_target_low_inputs,manager)
    cache_by_dataset={}
    offline_prompt_cache_by_dataset={}
    if offline_prompt_map:
        import hashlib
        prompt_source_path=ROOT/'artifacts/offline_force_baseline/train37_budget60/manifest.json'
        geometry_config_path=ROOT/'artifacts/ceer_segmented_jacobian/geometry.json'
        if offline_prompt_map.get('source_prompt_cache_manifest_sha256')!=digest(prompt_source_path):
            raise ValueError('Offline prompt map was built from a different source prompt configuration')
        if offline_prompt_map.get('geometry_sha256')!=digest(geometry_config_path):
            raise ValueError('Offline prompt map geometry configuration changed')
        if offline_prompt_map.get('manifest_sha256')!=digest(a.manifest):
            raise ValueError('Offline force-prompt cache was prepared for a different compliance manifest')
        for dataset,description in offline_prompt_map['datasets'].items():
            source=(ROOT/'dataset'/dataset).resolve()
            if hashlib.sha256((source/'meta_motion.json').read_bytes()).hexdigest()!=description['dataset_meta_sha256']:
                raise ValueError(f'Offline force-prompt cache dataset metadata changed: {dataset}')
            blob=np.load(description['path'])
            entries={}
            for mid,start,length,offset in zip(blob['motion_ids'],blob['start_steps'],blob['frames'],blob['offsets']):
                entries.setdefault(int(mid),[]).append((int(start),int(length),int(offset)))
            offline_prompt_cache_by_dataset[dataset]=dict(
                entries=entries,forces=torch.as_tensor(blob['forces'],device=device),
                sha256=hashlib.sha256(Path(description['path']).read_bytes()).hexdigest())
        for row in rows:
            choices=offline_prompt_cache_by_dataset.get(row['dataset'],{}).get('entries',{}).get(int(row['source_motion_id']),[])
            start_step=int(row['source_start_step']);last_step=start_step+int(row['frames'])-2
            if not any(start<=start_step and last_step<start+length for start,length,_ in choices):
                raise ValueError(f'No complete offline force-prompt coverage for {row}')
        def source_prompt_solve():
            cmd.point_force_buffer.zero_()
            if not cmd.guidance_enabled:return
            mids=cmd._motion.motion_id[:,0].long().tolist()
            steps_now=cmd._motion.step[:,0].long().tolist()
            for i,row in enumerate(rows):
                cache=offline_prompt_cache_by_dataset[row['dataset']]
                mid=int(mids[i]);step=int(steps_now[i]);index=None
                choices=cache['entries'].get(mid,[])
                for start,length,offset in choices:
                    if start<=step<start+length:
                        index=offset+step-start;break
                # The final reference frame has no next-step delta-q target;
                # OfflineForcePromptCommand uses a zero prompt there.
                if index is None and any(step==start+length for start,length,_ in choices):
                    continue
                if index is None:
                    raise RuntimeError(f'Offline force-prompt cache lost coverage: dataset={row["dataset"]} motion_id={mid} step={step}')
                cmd.point_force_buffer[i].copy_(cache['forces'][index])
        cmd._solve=source_prompt_solve
    if guidance_cache_map:
        # Resolve forces in the dataset namespace because motion_id values can collide.
        import hashlib
        max_motion_id=max(int(row['source_motion_id']) for row in rows)
        union_counts=torch.ones(max_motion_id+1,dtype=torch.long,device=device)
        geometry_hash=hashlib.sha256(Path(robot.cfg.spawn.usd_path).read_bytes()).hexdigest()
        for dataset,cache_paths in guidance_cache_map.items():
            if isinstance(cache_paths,str):cache_paths=[cache_paths]
            entries=[]
            for raw_path in cache_paths:
                path=Path(raw_path).resolve();cache_manifest=json.loads((path/'manifest.json').read_text())
                if [p['name'] for p in cache_manifest['points']]!=cmd.point_names:
                    raise ValueError(f'Force cache point layout mismatch for {dataset}: {path}')
                if cache_manifest['geometry_sha256']!=geometry_hash:
                    raise ValueError(f'Force cache geometry mismatch for {dataset}: {path}')
                offsets=torch.as_tensor(np.load(path/'motion_offsets.npy'),device=device).long()
                variant_slots=path/'variant_slots.npy';variant_forces=path/'variant_forces.npy'
                if variant_slots.exists() and variant_forces.exists():
                    slots=torch.as_tensor(np.load(variant_slots),device=device).long()
                    forces=torch.as_tensor(np.load(variant_forces),device=device)
                    counts=torch.as_tensor(np.load(path/'motion_variant_counts.npy'),device=device).long()
                else:
                    slots=torch.as_tensor(np.load(path/'slots.npy'),device=device).long().unsqueeze(0)
                    forces=torch.as_tensor(np.load(path/'forces.npy'),device=device).unsqueeze(0)
                    counts=torch.ones(len(offsets),dtype=torch.long,device=device)
                valid=torch.where((offsets>=0)&(counts>0))[0]
                valid=valid[valid<=max_motion_id]
                if valid.numel():union_counts[valid]=torch.maximum(union_counts[valid],counts[valid])
                entries.append(dict(path=str(path),manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),offsets=offsets,slots=slots,forces=forces,counts=counts))
            cache_by_dataset[dataset]=entries
        for row in rows:
            if row['dataset'] not in cache_by_dataset:
                raise ValueError(f'No source-specific guidance cache listed for {row["dataset"]}')
            mid=int(row['source_motion_id']);step=int(row['source_start_step']);frames=int(row['frames'])
            covered=any(mid<len(c['offsets']) and int(c['offsets'][mid])>=0 and int(c['offsets'][mid])+step+frames<=c['slots'].shape[1] for c in cache_by_dataset[row['dataset']])
            if not covered:raise ValueError(f'No complete source-specific force cache for {row["dataset"]} motion_id={mid} step={step} frames={frames}')
        cmd.cache_variant_counts=union_counts
        def source_solve():
            mids=cmd._motion.motion_id[:,0].long();steps_now=cmd._motion.step[:,0].long()
            cmd.point_desired.zero_()
            for i,row in enumerate(rows):
                mid=int(mids[i]);step=int(steps_now[i]);chosen=None
                for cache in cache_by_dataset[row['dataset']]:
                    if mid<len(cache['offsets']):
                        offset=int(cache['offsets'][mid])
                        if offset>=0 and offset+step<cache['slots'].shape[1]:chosen=(cache,offset+step);break
                if chosen is None:raise RuntimeError(f'Guidance cache lost source coverage: {row["dataset"]} motion_id={mid} step={step}')
                cache,index=chosen;variant=int(cmd.variant_ids[i])%int(cache['counts'][mid])
                slots=cache['slots'][variant,index]
                cmd.active[i].copy_(slots>=0);cmd.slots[i].copy_(slots.clamp_min(0))
                if cmd.guidance_enabled:cmd.desired[i].copy_(cache['forces'][variant,index])
                else:cmd.desired[i].zero_()
            cmd.point_desired.scatter_add_(1,cmd.slots[...,None].expand_as(cmd.desired),cmd.desired)
        cmd._solve=source_solve
    cmd.zero_init_prob=1.
    for key in cmd.init_noise_params:cmd.init_noise_params[key]=0.
    if a.suite=='tracking':
        ds=cmd.dataset
        for i,row in enumerate(rows):
            source=ds.datasets[paths.index(row['dataset'])];ci=row['clip'];st=int(source.starts[ci]);end=st+row['frames']
            idx=torch.arange(st,st+ds.max_step_size,device=source.starts.device).clamp_max(end-1)
            ds._buf_A[i]=ds._to_float(source.data[idx].to(device),dtype=ds.buffer_float_dtype);ds._len_A[i]=row['frames']
            for key in ds.dataset_extra_keys:ds._info_A[key['name']][i]=source.info[key['name']][ci].to(device)
    else:
        # Compliance behavior trials apply physical perturbations by default.
        # Offline prompt-only trials instead expose the same force vectors in
        # the actor's trained prompt interface, with simulator dispatch off.
        if hasattr(cmd,'force_apply_world'):cmd.force_apply_world=not a.prompt_only_behavior
        original=cmd.reset
        def standing(ids):
            original(ids)
            root=robot.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids];root[:,3:7]=root.new_tensor([1.,0,0,0]);root[:,7:]=0
            q=robot.data.default_joint_pos[ids].clone()
            # Same perturbation for every case with the same repeat, including its zero-force pair.
            for k,idx in enumerate(ids.tolist()):
                repeat=rows[idx%len(rows)]['repeat'];rng=np.random.default_rng(20261104+repeat)
                q[k]+=q.new_tensor(rng.normal(0,.003,q.shape[-1]))
            robot.write_root_state_to_sim(root,env_ids=ids);robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
            cmd.joint_pos_boot_protect[ids]=q
            for key in ('force','desired','force_apply_buffer','torque_apply_buffer'):
                if hasattr(cmd,key):getattr(cmd,key)[ids]=0
            torch.manual_seed(seed)
        callbacks=env.base_env._reset_callbacks;callbacks[:]=[standing if cb==original else cb for cb in callbacks];assert standing in callbacks
        cmd.before_update=lambda:setattr(cmd,'log_step',cmd.log_step+1)
        cmd.update=lambda:None
        body_ids=torch.zeros(n,2,dtype=torch.long,device=device);vectors=torch.zeros(n,2,3,device=device);offsets=torch.zeros(n,2,3,device=device)
        torso_id=robot.body_names.index('torso_link')
        for i in range(n):
            row=rows[i%len(rows)]
            for j,(body,v) in enumerate(row['loads']):
                body_ids[i,j]=robot.body_names.index(body)
                offsets[i,j]=offsets.new_tensor(row.get('offsets',{}).get(body,[0.,0.,0.]))
                if not a.zero:vectors[i,j]=vectors.new_tensor(v)*row['force_N']
        clock_step=0
        applied=torch.zeros_like(vectors)
        def force_step(substep):
            t=clock_step*env.step_dt+substep*env.physics_dt;applied.copy_(vectors*envelope(t))
            if a.body_relative_bend:
                # Torso local +X remains perpendicular to its local +Z body axis.
                torso_forward=quat_apply(robot.data.body_quat_w[:,torso_id],vectors.new_tensor([1.,0.,0.]).expand(n,3))
                force_magnitude=vectors[:,0].norm(dim=-1)*envelope(t)
                applied[:,0]=torso_forward*force_magnitude[:,None]
                applied[:,1]=-torso_forward*force_magnitude[:,None]
            cmd.force_apply_buffer.zero_();cmd.torque_apply_buffer.zero_()
            cmd.force_apply_buffer.scatter_add_(1,body_ids[...,None].expand_as(applied),applied)
            # Offset-based generators keep a separate observation buffer. Populate only exact origin matches.
            if hasattr(cmd,'point_offsets'):
                cmd.point_force_buffer.zero_()
                for pi,bid in enumerate(cmd.point_ids.tolist()):
                    if float(cmd.point_offsets[pi].norm())<1e-8:cmd.point_force_buffer[:,pi]=((body_ids==bid)[...,None]*applied).sum(1)
            elif hasattr(cmd,'cache_manifest') and hasattr(cmd,'point_force_buffer'):
                # Offline force-prompt actors receive the applied GT force only
                # at body/offset points present in their trained 37-point input.
                cmd.point_force_buffer.zero_()
                for pi,point in enumerate(cmd.cache_manifest['points']):
                    bid=robot.body_names.index(point['body'])
                    point_offset=offsets.new_tensor(point['offset'])
                    matches=(body_ids==bid)&((offsets-point_offset).norm(dim=-1)<1e-7)
                    cmd.point_force_buffer[:,pi]=(matches[...,None]*applied).sum(1)
            if a.prompt_only_behavior:
                # Keep force traces and simulator buffers at zero. The prompt
                # lives only in point_force_buffer and is consumed by policy obs.
                cmd.force_apply_buffer.zero_();cmd.torque_apply_buffer.zero_()
            cmd.position_apply_buffer.copy_(robot.data.body_pos_w)
            for slot in range(2):
                mask=offsets[:,slot].norm(dim=-1)>0
                ids=torch.arange(n,device=device)[mask];bids=body_ids[mask,slot]
                cmd.position_apply_buffer[ids,bids]+=quat_apply(robot.data.body_quat_w[ids,bids],offsets[mask,slot])
            robot.has_external_wrench=False
        original=cmd.step;callbacks=env.base_env._pre_step_callbacks;callbacks[:]=[force_step if cb==original else cb for cb in callbacks];assert force_step in callbacks
    if a.policy_branch=='teacher':
        if phase!='train' or not hasattr(agent,'actor_teacher'):
            raise ValueError('--policy-branch teacher requires a RootStudentPPO teacher checkpoint with phase=train')
        policy=agent.get_rollout_policy('train')
    else:
        policy=agent.get_rollout_policy('deploy')
    supervised=None; supervised_blob=None; supervised_current_q=False
    if a.supervised_deltaq_checkpoint:
        supervised_blob=torch.load(a.supervised_deltaq_checkpoint,map_location='cpu',weights_only=False)
        from scripts.train_force_to_deltaq_residual import build_model
        history_offsets=list(map(int,supervised_blob.get('history_offsets_frames',[0])))
        force_input_dim=int(supervised_blob.get('force_input_dim',9*len(history_offsets)))
        supervised_type=supervised_blob.get('model_type','force_to_delta_q')
        supervised_current_q=supervised_type=='force_current_q_to_next_q'
        model_input_dim=force_input_dim+(29 if supervised_current_q else 0)
        supervised=build_model(input_dim=model_input_dim).to(device).eval();supervised.load_state_dict(supervised_blob['model'],strict=True)
        if hasattr(cmd,'point_names'):
            point_names=list(cmd.point_names)
        else:
            point_names=[robot.body_names[int(i)] for i in cmd.point_ids]
        if point_names!=list(supervised_blob['point_names']):
            raise RuntimeError(f'Supervised force point order mismatch: env={point_names}, model={list(supervised_blob["point_names"])}')
        model_joint_names=list(supervised_blob['joint_names'])
        manager_joint_names=list(manager.joint_names)
        motion_joint_order=[cmd.dataset.joint_names.index(name) for name in manager_joint_names]
        if len(model_joint_names)!=len(manager_joint_names) or set(model_joint_names)!=set(manager_joint_names):
            raise RuntimeError(f'Supervised delta-q joints differ from reference interface: model={model_joint_names}, reference={manager_joint_names}')
        supervised_joint_order=torch.tensor([model_joint_names.index(name) for name in manager_joint_names],device=device)
        manager_to_model_joint_order=torch.tensor([manager_joint_names.index(name) for name in model_joint_names],device=device)
        if not hasattr(manager,'reference') or not hasattr(manager.reference,'speed'):
            raise RuntimeError('Supervised delta-q adapter requires the CEERResidualAction reference interface')
        if supervised_current_q:
            input_mean=torch.as_tensor(supervised_blob['input_mean'],device=device,dtype=torch.float32)
            input_std=torch.as_tensor(supervised_blob['input_std'],device=device,dtype=torch.float32)
            output_mean=torch.as_tensor(supervised_blob['output_mean_rad'],device=device,dtype=torch.float32)
            output_scale=torch.as_tensor(supervised_blob['output_std_rad'],device=device,dtype=torch.float32)
        else:
            input_mean=torch.as_tensor(supervised_blob['input_mean_N'],device=device,dtype=torch.float32)
            input_std=torch.as_tensor(supervised_blob['input_std_N'],device=device,dtype=torch.float32)
            output_scale=torch.as_tensor(supervised_blob['output_scale_rad'],device=device,dtype=torch.float32)
        force_values_per_frame=len(point_names)*3
        if input_mean.numel()!=model_input_dim or len(history_offsets)*force_values_per_frame!=force_input_dim:
            raise RuntimeError(f'Unexpected supervised force feature schema: dim={force_input_dim}, offsets={history_offsets}')
        force_history_buffer=torch.zeros(n,max(history_offsets)+1,force_values_per_frame,device=device)
        actor_inputs=[f'force_gt_ history offsets {history_offsets} ({len(point_names)} points, N, root frame)']
        if supervised_current_q: actor_inputs.append(f'current {a.supervised_q_input_source} 29-joint positions (rad)')
    else:
        if a.policy_branch=='deploy':
            assert not set(policy.in_keys)&{'joint_target','priv','command'},f'Target input not allowed: {policy.in_keys}'
        else:
            assert {'joint_target','priv'}.issubset(set(policy.in_keys)),f'Oracle teacher inputs missing: {policy.in_keys}'
        actor_inputs=list(policy.in_keys)
    torch.manual_seed(seed);td=env.reset();env.base_env.episode_length_buf.zero_()
    if a.suite=='tracking':
        for i,row in enumerate(rows):
            assert int(cmd._motion.motion_id[i,0])==row['source_motion_id']
            assert int(cmd._motion.step[i,0])==row['source_start_step']
    elif a.suite=='behavior':
        low=getattr(manager,'low_action_manager',manager)
        if hasattr(low,'alpha_jit_scale'):low.alpha_jit_scale=None
        if hasattr(low,'alpha'):low.alpha.fill_(sum(low.alpha_range)/2)
        if hasattr(low,'delay'):low.delay.fill_((low.min_delay+low.max_delay)//2)
        if hasattr(low,'boot_delay'):low.boot_delay.copy_(low.delay)
    initial_q=robot.data.joint_pos.clone();initial_root=(robot.data.root_pos_w-env.scene.env_origins).clone()
    hands=[robot.body_names.index(f'{s}_hand_mimic') for s in ('left','right')];torso=robot.body_names.index('torso_link')
    motionhands=[cmd.dataset.body_names.index(f'{s}_hand_mimic') for s in ('left','right')]
    joint_groups={}
    for group in ['left_arm','right_arm','legs','waist']:
        joint_groups[group]=[i for i,j in enumerate(cmd.joint_idx_asset.tolist()) if
            (group=='legs' and any(t in robot.joint_names[j] for t in ('hip','knee','ankle'))) or
            (group=='waist' and 'waist' in robot.joint_names[j]) or
            (group.endswith('_arm') and robot.joint_names[j].startswith(group.split('_')[0]) and any(t in robot.joint_names[j] for t in ('shoulder','elbow','wrist')))]
    steps=(max(r['frames'] for r in rows)-1) if a.suite=='tracking' else 600
    if a.steps:steps=min(steps,a.steps)
    record={};actor_prompt_input=[];first_fall=torch.full((n,),-1,dtype=torch.int32,device=device);valid=[]
    estimator_audit_state=None
    if a.estimator_audit:
        if a.suite!='tracking' or offline_prompt_map is None:
            raise ValueError('--estimator-audit requires tracking with --offline-prompt-cache-map')
        if getattr(agent,'adapt_direct_module',None) is None or 'joint_target' not in agent.direct_priv_keys:
            raise ValueError('Checkpoint has no direct joint_target estimator')
        estimator_audit_state={'frames':0,'abs_sum':None,'sq_sum':None,'base_abs_sum':None,'base_sq_sum':None,
                               'force_abs_max_N':0.0,'force_sq_sum':0.0,'force_values':0,'force_nonzero_frames':0,
                               'force_frames':0,'force_max_N':0.0}
    prev_force=cmd.force_apply_buffer.clone();prev_q=robot.data.joint_pos.clone();prev_ref=cmd._motion.joint_pos[:,0,cmd.joint_idx_motion].clone()
    if video:
        from scripts.compliance_eval.visual import Video
        video_path=a.video_output or a.output/'rollout.mp4'
        video_path.parent.mkdir(parents=True,exist_ok=True)
        renderer=Video(env,cmd,video_path,a.suite=='tracking',
                       prompt_only=(bool(offline_prompt_map) and not getattr(cmd,'physical_guidance',False)) or a.prompt_only_behavior)
    def save(k,v):record.setdefault(k,[]).append(v.detach().cpu().numpy().astype(np.float32))
    with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
        for step in range(steps+1):
            if step:
                clock_step=step-1
                if supervised is not None:
                    force_now=td['force_gt_'].reshape(n,-1).float()*30.0
                    if force_now.shape[-1]!=force_history_buffer.shape[-1]:
                        raise RuntimeError(f'Supervised model force input shape mismatch: obs={tuple(force_now.shape)}, expected={force_history_buffer.shape[-1]} values')
                    force_history_buffer[:,1:]=force_history_buffer[:,:-1].clone()
                    force_history_buffer[:,0]=force_now
                    force_features=torch.cat([force_history_buffer[:,offset] for offset in history_offsets],dim=-1)
                    model_features=force_features
                    if supervised_current_q:
                        current_q_manager=(robot.data.joint_pos[:,manager.joint_ids]
                                           if a.supervised_q_input_source=='measured'
                                           else manager.reference.q)
                        current_q_model=current_q_manager[:,manager_to_model_joint_order]
                        model_features=torch.cat((force_features,current_q_model),dim=-1)
                    pred=supervised((model_features-input_mean)/input_std)
                    if supervised_current_q:
                        pred=pred*output_scale+output_mean
                        pred=pred[:,supervised_joint_order]
                        if a.suite=='tracking':
                            # Audit the deployed measured-state prediction against
                            # the next reference frame at the same rollout time.
                            next_reference=cmd.dataset.get_slice(None,cmd.t+1,1).joint_pos[:,0,motion_joint_order]
                            next_error=pred-next_reference
                            save('predicted_next_q_reference_mae_rad',next_error.abs().mean(-1))
                            save('predicted_next_q_reference_rmse_rad',next_error.square().mean(-1).sqrt())
                        delta_q=pred-manager.reference.q
                    else:
                        pred=pred*output_scale
                        pred=pred[:,supervised_joint_order]
                        delta_q=pred
                    ratio=(delta_q/(env.step_dt*manager.reference.speed)).clamp(-.95,.95)
                    td['action']=torch.atanh(ratio)
                    if a.zero_residual:td['action'].zero_()
                    save('predicted_delta_q_rms_rad',delta_q.square().mean(-1).sqrt())
                    save('predicted_delta_q_saturated_fraction',(delta_q.abs()>env.step_dt*manager.reference.speed*.95).float().mean(-1))
                    if supervised_current_q: save('predicted_next_q_rms_rad',pred.square().mean(-1).sqrt())
                    td=env.step(td)['next']
                else:
                    if a.ceer_direct_joint_target:
                        # Feed the current motion frame into the CEER reference
                        # interface; the low-level target tensor is replaced
                        # above by the actual five reference frames.
                        manager.reference.q.copy_(cmd._motion.joint_pos[:,0,motion_joint_ids])
                        td['action']=torch.zeros(n,manager.action_dim,device=device)
                        td=env.step(td)['next']
                        policy_output_bypassed=True
                    else:
                        policy_output_bypassed=False
                    if policy_output_bypassed:
                        pass
                    else:
                        if a.suite=='tracking' and 'force_gt_' in td and 'force_gt_' in policy.in_keys:
                        # Audit the exact prompt tensor seen by the deployed
                        # actor immediately before its action is computed.
                            prompt=td['force_gt_'].reshape(n,-1,3).float()*30.0
                            actor_prompt_input.append(prompt.norm(dim=-1).sum(-1).detach().cpu().numpy())
                        td=policy(td)
                        if estimator_audit_state is not None:
                            if 'direct_priv_pred' not in td.keys() or 'joint_target' not in td.keys():
                                raise RuntimeError('Deployed policy did not expose direct_priv_pred and joint_target')
                            pred=td['direct_priv_pred'].reshape(n,-1).float()
                            target=td['joint_target'].reshape(n,-1).float()
                            if pred.shape!=target.shape:
                                raise RuntimeError(f'Estimator/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}')
                            joint_names=list(cmd.dataset.joint_names)
                            joint_count=len(joint_names)
                            if pred.shape[-1]!=joint_count:
                                raise RuntimeError(f'Unexpected joint_target layout: dim={pred.shape[-1]}, joints={joint_count}')
                            mask=(first_fall<0)&(step<cmd.lengths)
                            if mask.any():
                                target_phys=target*target_scale+target_loc
                                pred_phys=pred*target_scale+target_loc
                                err=(pred_phys-target_phys)[mask].double().cpu()
                                if estimator_audit_state['abs_sum'] is None:
                                    estimator_audit_state.update({
                                        'joint_names':joint_names,
                                        'abs_sum':torch.zeros(joint_count,dtype=torch.float64),
                                        'sq_sum':torch.zeros(joint_count,dtype=torch.float64),
                                        'base_abs_sum':torch.zeros(joint_count,dtype=torch.float64),
                                        'base_sq_sum':torch.zeros(joint_count,dtype=torch.float64),
                                    })
                                estimator_audit_state['abs_sum']+=err.abs().sum(0)
                                estimator_audit_state['sq_sum']+=err.square().sum(0)
                                estimator_audit_state['frames']+=err.shape[0]
                                asset_ids=[robot.joint_names.index(name) for name in joint_names]
                                current_q=robot.data.joint_pos[mask][:,asset_ids].double().cpu()
                                baseline_err=current_q-target_phys[mask].double().cpu()
                                estimator_audit_state['base_abs_sum']+=baseline_err.abs().sum(0)
                                estimator_audit_state['base_sq_sum']+=baseline_err.square().sum(0)
                            observed=td['force_gt_'].reshape(n,-1,3).float()*30.0
                            point_count=len(cmd.point_names)
                            expected=quat_apply_inverse(
                                robot.data.root_quat_w[:,None].expand(-1,point_count,-1),
                                cmd.point_force_buffer,
                            )
                            force_err=(observed-expected).abs()
                            force_mask=(first_fall<0)&(step<cmd.lengths)
                            if force_mask.any():
                                selected=force_err[force_mask]
                                norms=observed[force_mask].norm(dim=-1)
                                estimator_audit_state['force_abs_max_N']=max(estimator_audit_state['force_abs_max_N'],float(selected.max().item()))
                                estimator_audit_state['force_sq_sum']+=float(selected.double().square().sum().item())
                                estimator_audit_state['force_values']+=selected.numel()
                                estimator_audit_state['force_nonzero_frames']+=int((norms.sum(-1)>1e-5).sum().item())
                                estimator_audit_state['force_frames']+=int(force_mask.sum().item())
                                estimator_audit_state['force_max_N']=max(estimator_audit_state['force_max_N'],float(norms.max().item()))
                        if a.zero_residual:td['action'].zero_()
                        td=env.step(td)['next']
            root=robot.data.root_pos_w-env.scene.env_origins;q=robot.data.joint_pos;quat=robot.data.root_quat_w
            tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1));fall=(root[:,2]<.30)|(tilt>np.deg2rad(75))|(~torch.isfinite(q).all(-1))
            first_fall=torch.where((first_fall<0)&fall,torch.full_like(first_fall,step),first_fall)
            active=first_fall<0
            if a.suite=='tracking':active &= step<cmd.lengths
            valid.append(active.cpu().numpy())
            h=robot.data.body_pos_w[:,hands]-robot.data.root_pos_w[:,None]
            hb=quat_apply_inverse(quat[:,None].expand(-1,2,-1),h)
            rel=quat_mul(quat_conjugate(quat),robot.data.body_quat_w[:,torso])
            forces=cmd.force_apply_buffer
            save('root',root);save('yaw',yaw_from_quat(quat));save('hands_b',hb);save('torso_relative_quat',rel);save('root_velocity',robot.data.root_lin_vel_w)
            if a.suite=='tracking':
                save('reference_root',cmd._motion.root_pos_w[:,0]-env.scene.env_origins)
                save('motion_step',cmd._motion.step[:,0]);save('motion_id',cmd._motion.motion_id[:,0])
                save('episode_length',env.base_env.episode_length_buf)
            if hasattr(manager,'reference'):
                save('reference_offset_rad',(manager.reference.q-initial_q[:,manager.joint_ids]).abs().mean(-1))
                save('reference_tracking_gap_rad',(manager.reference.q-q[:,manager.joint_ids]).abs().mean(-1))
                save('residual_action_norm',manager.applied_action.norm(dim=-1))
            save('joint_pos',q);save('force_sum_N',forces.norm(dim=-1).sum(-1));save('force_peak_N',forces.norm(dim=-1).amax(-1));save('active_points',(forces.norm(dim=-1)>.05).sum(-1))
            save('force_slew_N_s',(forces-prev_force).norm(dim=-1).amax(-1)/env.step_dt);prev_force.copy_(forces)
            save('pelvis_force_N',forces[:,robot.body_names.index('pelvis')].norm(dim=-1));save('net_force',forces.sum(1));save('net_moment',(torch.cross(cmd.position_apply_buffer-robot.data.root_pos_w[:,None],forces,dim=-1)+cmd.torque_apply_buffer).sum(1))
            if a.suite=='tracking':
                refq=cmd._motion.joint_pos[:,0,cmd.joint_idx_motion];dq=(q[:,cmd.joint_idx_asset]-refq).abs()
                save('joint_mae_rad',dq.mean(-1))
                for g,ids in joint_groups.items():save(g+'_mae_rad',dq[:,ids].mean(-1))
                rp=cmd._motion.root_pos_w[:,0];rq=cmd._motion.root_quat_w[:,0];refh=cmd._motion.body_pos_w[:,0,motionhands]
                refhb=quat_apply_inverse(rq[:,None].expand(-1,2,-1),refh-rp[:,None])
                save('ee_world_error_m',(robot.data.body_pos_w[:,hands]-env.scene.env_origins[:,None]-refh).norm(dim=-1))
                save('ee_relative_error_m',(hb-refhb).norm(dim=-1))
                save('root_xy_error_m',(root[:,:2]-rp[:,:2]).norm(dim=-1));save('root_z_error_m',(root[:,2]-rp[:,2]).abs())
                yd=yaw_from_quat(quat)-yaw_from_quat(rq);save('yaw_error_rad',torch.atan2(yd.sin(),yd.cos()).abs())
                for side in ('left_arm','right_arm'):
                    ids=joint_groups[side];stationary=((refq-prev_ref)[:,ids].abs().amax(-1)/env.step_dt)<.05
                    save(side+'_stationary',stationary);save(side+'_stationary_motion_rad_s',((q-prev_q)[:,cmd.joint_idx_asset][:,ids].abs().mean(-1)/env.step_dt))
                prev_ref.copy_(refq)
            prev_q.copy_(q)
            if video and step%2==0:
                renderer.frame(step*env.step_dt,rows[0].get('name',rows[0].get('category')),first_fall[0].item()>=0)
            if step in (50,200,500,steps):print(json.dumps(dict(step=step,total=steps,elapsed_s=time.monotonic()-started,envs=n)),flush=True)
    if video:renderer.close()
    arrays={k:np.stack(v) for k,v in record.items()};arrays['valid']=np.stack(valid);arrays['first_fall_step']=first_fall.cpu().numpy();arrays['initial_joint_pos']=initial_q.cpu().numpy();arrays['initial_root']=initial_root.cpu().numpy()
    if actor_prompt_input:arrays['actor_prompt_input_sum_N']=np.stack(actor_prompt_input)
    mismatch=None
    np.savez_compressed(a.output/'traces.npz',**arrays)
    points=(list(cmd.point_names) if hasattr(cmd,'point_names') else
            [robot.body_names[int(i)] for i in cmd.point_ids] if hasattr(cmd,'point_ids') else [])
    result=dict(version=manifest['version'],suite=a.suite,policy_branch=a.policy_branch,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest(a.checkpoint),manifest_sha256=digest(a.manifest),
                zero_residual_control=a.zero_residual,zero_force_control=a.zero,seed=seed,rows=rows,envs=n,steps=steps,dt=float(env.step_dt),wall_seconds=time.monotonic()-started,phase=phase,
                force_history_offsets=history_offsets if supervised is not None else getattr(cmd,'force_prompt_history_offsets',None),force_input_bodies=points,paired_preforce_max_difference_m=mismatch,
                prompt_only_behavior=bool(a.prompt_only_behavior),preserve_training_sim=bool(a.preserve_training_sim),physical_force_dispatch=bool(getattr(cmd,'force_apply_world',False)),
                joint_names=robot.joint_names,configuration=saved_config,video=video,actor_inputs=(['direct full-body joint targets at reference offsets [0,2,4,8,16]'] if a.ceer_direct_joint_target else actor_inputs),
                direct_ceer_joint_target=a.ceer_direct_joint_target,
                supervised_deltaq_checkpoint=str(a.supervised_deltaq_checkpoint.resolve()) if a.supervised_deltaq_checkpoint else None,
                supervised_deltaq_checkpoint_sha256=digest(a.supervised_deltaq_checkpoint) if a.supervised_deltaq_checkpoint else None,
                supervised_q_input_source=a.supervised_q_input_source if supervised_current_q else None,
                action_adapter=('absolute next-step joint target - persistent SONIC reference -> inverse-tanh increment' if supervised_current_q else 'physical next-step delta-q -> inverse-tanh reference increment') if supervised is not None else None)
    result['guidance_cache_map']={dataset:[dict(path=c['path'],manifest_sha256=c['manifest_sha256']) for c in caches] for dataset,caches in cache_by_dataset.items()} if guidance_cache_map else None
    result['guidance_cache_map_sha256']=digest(a.guidance_cache_map) if a.guidance_cache_map else None
    result['offline_prompt_cache_map']=offline_prompt_map if a.offline_prompt_cache_map else None
    result['offline_prompt_cache_map_sha256']=digest(a.offline_prompt_cache_map) if a.offline_prompt_cache_map else None
    if estimator_audit_state is not None:
        audit=estimator_audit_state
        def summarize(abs_sum,sq_sum,frames):
            mae=abs_sum/frames;rmse=(sq_sum/frames).sqrt()
            return {'mae_rad':float(abs_sum.sum().item()/(frames*abs_sum.numel())),
                    'rmse_rad':float((sq_sum.sum().item()/(frames*sq_sum.numel()))**.5),
                    'by_joint':{name:{'mae_rad':float(mae[j].item()),'rmse_rad':float(rmse[j].item())} for j,name in enumerate(audit['joint_names'])}}
        estimate=summarize(audit['abs_sum'],audit['sq_sum'],audit['frames'])
        baseline=summarize(audit['base_abs_sum'],audit['base_sq_sum'],audit['frames'])
        force_rms=(audit['force_sq_sum']/max(audit['force_values'],1))**.5
        estimator_report={'target':'joint_target; 29D current reference joint positions (inverse VecNorm applied; radians)','frames':audit['frames'],
                          'joint_names':audit['joint_names'],
                          'estimate':estimate,'current_measured_q_baseline':baseline,
                          'force_token_consistency':{'comparison':'force_gt_ * 30 N vs current cached point_force_buffer transformed into root frame',
                              'max_abs_error_N':audit['force_abs_max_N'],'rmse_N':force_rms,
                              'max_observed_point_force_N':audit['force_max_N'],
                              'nonzero_prompt_frame_fraction':audit['force_nonzero_frames']/max(audit['force_frames'],1)}}
        result['estimator_audit']=estimator_report
        (a.output/'estimator_audit.json').write_text(json.dumps(estimator_report,indent=2))
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print('EVAL_DONE',a.output,flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(1)
else:
    # Isaac 4.x shutdown can hang after results are flushed. This dedicated
    # subprocess owns all GPU/simulator resources; OS exit releases them.
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)
