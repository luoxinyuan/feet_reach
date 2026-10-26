"""Audit static-target teacher: FK consistency, target conditioning and joint errors."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE','1')
import torch
from omegaconf import OmegaConf
import active_adaptation.learning
from isaaclab.app import AppLauncher
run=Path((ROOT/'outputs/passive-ee-amass/latest_pipeline.txt').read_text().strip());ckpt=next((run/'train').rglob('checkpoint_final.pt'))
s=torch.load(ckpt,map_location='cpu',weights_only=False);cfg=OmegaConf.create(OmegaConf.to_container(s['cfg'],resolve=True));del s
OmegaConf.set_struct(cfg,False);cfg.checkpoint_path=str(ckpt);cfg.vecnorm='eval';cfg.task.num_envs=256;cfg.seed=42
cfg.task.command.dataset.mem_paths=['passive_ee_amass/val'];cfg.task.command.hold_probability=1.;cfg.wandb.mode='disabled'
app=AppLauncher(headless=True,device='cuda:0').app
try:
 from scripts.utils.helpers import make_env_policy
 from torchrl.envs.utils import set_exploration_type,ExplorationType
 torch.manual_seed(42);env,agent,_,_=make_env_policy(cfg);env.eval();env.base_env.eval();agent.eval();cmd=env.command_manager;asset=cmd.asset
 cmd.zero_init_prob=1.;torch.manual_seed(42);carry=env.reset();policy=agent.get_rollout_policy('train')
 names=[asset.joint_names[i] for i in cmd.joint_idx_asset.tolist()];arm=[i for i,n in enumerate(names) if any(k in n for k in ['shoulder','elbow','wrist'])];leg=[i for i,n in enumerate(names) if any(k in n for k in ['hip','knee','ankle'])]
 records=[]
 def measure(step):
  err=(asset.data.joint_pos[:,cmd.joint_idx_asset]-cmd._motion.joint_pos[:,0,cmd.joint_idx_motion]).abs()
  body=(asset.data.body_pos_w[:,cmd.keypoint_idx_asset]-cmd.reward_keypoints_w[:,cmd.keypoint_idx_motion]).norm(dim=-1)
  ee=(asset.data.body_pos_w[:,cmd.hand_ids]-cmd.anchor).norm(dim=-1)
  return dict(step=step,time_s=step*.02,joint_mae=float(err.mean()),arm_mae=float(err[:,arm].mean()),leg_mae=float(err[:,leg].mean()),ee_error=float(ee.mean()),keypoint_error=float(body.mean()),per_joint_mae=dict(zip(names,err.mean(0).tolist())))
 with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
  records.append(measure(0))
  base=carry.clone();policy(base);shuffled=carry.clone();shuffled['joint_target']=shuffled['joint_target'].roll(1,0);shuffled['priv']=shuffled['priv'].roll(1,0);policy(shuffled)
  sensitivity=dict(teacher_in_keys=[str(k) for k in policy.in_keys],action_difference_from_shuffled_targets=float((base['loc']-shuffled['loc']).abs().mean()))
  for step in range(1,251):
   carry=policy(carry);td=env.step(carry);carry=td['next']
   if step in [1,10,50,100,250]:records.append(measure(step))
 report=dict(checkpoint=str(ckpt),scenario='fixed reference, zero force, no resets; all 256 included for FK/conditioning audit',sensitivity=sensitivity,records=records)
 out=ROOT/'artifacts/passive_ee_amass_evaluation_20261021/teacher_diagnostic.json';out.write_text(json.dumps(report,indent=2));print('DIAGNOSTIC_OK',json.dumps(report),flush=True)
 from isaaclab.sim import SimulationContext
 SimulationContext.instance()._disable_app_control_on_stop_handle=True;env.close()
except BaseException:
 import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
