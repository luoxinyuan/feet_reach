"""Check cache provenance, boundaries, physical limits and random exact deltas."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.dense_link_guidance import cap_vectors
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--full-hash',action='store_true');a=p.parse_args()
m=json.loads((a.cache/'manifest.json').read_text());source=Path(m['source_dataset'])
assert hashlib.sha256((source/'meta_motion.json').read_bytes()).hexdigest()==m['source_meta_sha256']
f=np.load(a.cache/'forces.npy',mmap_mode='r');valid=np.load(a.cache/'valid_next.npy');offset=np.load(a.cache/'motion_offsets.npy')
ds=MotionDataset.create_from_path_lazy(str(source));ids=[ds.body_names.index(n) for n in m['body_names']]
assert f.shape==(len(ds.data),len(ids),3)
assert np.max(np.abs(f[ds.ends.numpy()-1]))==0
rng=np.random.default_rng(20261027);idx=rng.integers(0,len(f)-1,8192)
x=ds.data.body_pos_w[torch.from_numpy(idx)][:,ids].float().numpy()
y=ds.data.body_pos_w[torch.from_numpy(idx+1)][:,ids].float().numpy()
expected=cap_vectors((y-x)*valid[idx,None,None]*m['k_N_per_m'])
np.testing.assert_allclose(f[idx],expected,atol=1e-6)
mid=ds.data.motion_id[idx].numpy().reshape(-1);step=ds.data.step[idx].numpy().reshape(-1)
np.testing.assert_array_equal(offset[mid]+step,idx)
assert m['force_metrics']['point_max']['max']<=30.00001
for lo in range(0,len(f),65536):
 chunk=np.asarray(f[lo:lo+65536]);assert np.isfinite(chunk).all();assert np.linalg.norm(chunk,axis=-1).max()<=30.00001
if a.full_hash:
 for name,digest in m['files_sha256'].items():
  h=hashlib.sha256()
  with (a.cache/name).open('rb') as file:
   for chunk in iter(lambda:file.read(8*1024**2),b''):h.update(chunk)
  assert h.hexdigest()==digest,name
print('DENSE_CACHE_VALID',a.cache,len(f),len(ids),m['k_N_per_m'])
