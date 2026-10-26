"""Reuse selected AMASS moving clips and create separate stationary hold clips."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active_adaptation.utils.motion import MotionDataset
torch.set_num_threads(2)
out=ROOT/'dataset/passive_ee_continuous'
if out.exists():raise FileExistsError(out)
for split in ['train','val']:
 source=ROOT/f'dataset/passive_ee_amass/{split}';ds=MotionDataset.create_from_path_lazy(str(source))
 for kind in ['move','hold']:
  dest=out/split/kind;dest.mkdir(parents=True);data=ds.data.clone()
  if kind=='hold':
   for start,end in zip(ds.starts.tolist(),ds.ends.tolist()):
    index=int(data.joint_vel[start:end].float().square().mean(-1).argmin())+start
    for field in data.__dataclass_fields__:
     if field in ['motion_id','step']:continue
     value=getattr(data,field);value[start:end]=value[index].clone()
   data.joint_vel.zero_();data.root_lin_vel_w.zero_();data.root_ang_vel_w.zero_()
  data.memmap(str(dest));meta=json.loads((source/'meta_motion.json').read_text())
  meta['info']={'is_hold':[[float(kind=='hold')]]*ds.num_motions};(dest/'meta_motion.json').write_text(json.dumps(meta))
  print(split,kind,ds.num_motions,flush=True)
