"""Check the actual training reader against source FK and split metadata."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from active_adaptation.utils.motion import MotionDataset
from scipy.spatial.transform import Rotation
def quat_apply(q, v):
    return torch.from_numpy(Rotation.from_quat(q.numpy()[:,[1,2,3,0]]).apply(v.numpy()))
def quat_apply_inverse(q, v):
    return torch.from_numpy(Rotation.from_quat(q.numpy()[:,[1,2,3,0]]).inv().apply(v.numpy()))
source=ROOT/'artifacts/foot_reach'
manifest=json.loads((source/'manifest.json').read_text())
sets={s:{c['endpoint_id'] for c in manifest['clips'] if c['split']==s} for s in ('train','val')}
assert not sets['train']&sets['val']
for split in sets:
    ds=MotionDataset.create_from_path_lazy(str(ROOT/'dataset/wall_foot_reach'/split))
    clips=[c for c in manifest['clips'] if c['split']==split]
    assert ds.data.joint_pos.shape[0]==sum(c['frames'] for c in clips)
    assert len(ds.starts)==len(clips)
    assert ds.joint_names==manifest['joint_names'] and ds.body_names==manifest['body_names']
    for _,v in ds.data.items():
        if v.dtype.is_floating_point: assert torch.isfinite(v).all()
    order=json.loads((ROOT/"dataset/wall_foot_reach"/split/"source_files.json").read_text())
    by_file={c["file"]:c for c in clips}
    for i,path in enumerate(order):
        c=by_file[path]
        raw=np.load(source/c['file'])
        start=int(ds.starts[i]);n=c['frames']
        data=ds.data[start:start+n]
        np.testing.assert_allclose(data.joint_pos.numpy(),raw['dof_pos'],atol=1e-7)
        np.testing.assert_allclose(data.root_pos_w.numpy(),raw['root_pos'],atol=1e-7)
        np.testing.assert_allclose(data.body_pos_b.numpy(),raw['local_body_pos'],atol=2e-7)
        index=ds.body_names.index('left_ankle_roll_link')
        offset=torch.tensor(manifest['sole_offset'],dtype=torch.float32).expand(n,-1)
        sole=data.body_pos_w[:,index]+quat_apply(data.body_quat_w[:,index],offset)
        target=quat_apply_inverse(data.root_quat_w,sole-data.root_pos_w)
        np.testing.assert_allclose(target.numpy(),raw['target_foot_pos_b'],atol=2e-7)
        # C2 holds at both ends; full body except swing leg remains anchored.
        assert np.max(abs(raw['joint_vel'][[0,-1]]))<1e-6
        assert np.max(abs(raw['joint_acc'][[0,-1]]))<1e-6
        np.testing.assert_allclose(raw['dof_pos'][:,6:],np.broadcast_to(raw['dof_pos'][0,6:],raw['dof_pos'][:,6:].shape),atol=1e-7)
    print(split,len(clips),'clips: schema, FK xyz, fixed joints, finite values, C2 endpoints passed')
print('FOOT_REACH_DATASET_OK')
