"""Audit prompt bounds, temporal alignment, source copying, and dataset integrity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.compliance_eval.protocol import load_arrays

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args();m=json.loads((a.cache/'manifest.json').read_text());assert m['complete']
 cache=np.load(a.cache/'forces.npy',mmap_mode='r');offsets=np.load(a.cache/'motion_offsets.npy');packed=[load_arrays(Path(x))[1] for x in m['dataset_paths']];sources=[load_arrays(Path(x['path']))[1] for x in m['sources']]
 maxsum=maxpoint=maxslew=0.;rng=np.random.default_rng(20261105);chosen=set(rng.choice(len(m['clips']),min(100,len(m['clips'])),replace=False).tolist())
 for c in m['clips']:
  off=c['force_offset'];n=c['frames'];f=cache[off:off+n];assert np.isfinite(f).all();norm=np.linalg.norm(f.astype(float),axis=-1)
  maxsum=max(maxsum,float(norm.sum(1).max()));maxpoint=max(maxpoint,float(norm.max()));assert not f[-1].any()
  if n>2:maxslew=max(maxslew,float(np.linalg.norm(np.diff(f[:-1].astype(float),axis=0),axis=-1).max()/.02))
  assert offsets[c['id']]==off
  data=packed[c['source_index']];lo=c['start'];assert np.all(data['motion_id'][lo:lo+n]==c['id']);np.testing.assert_array_equal(data['step'][lo:lo+n],np.arange(n))
  if c['id'] in chosen:
   source=sources[c['source_index']];old=c['source_start']
   for key in data:
    if key not in ['motion_id','step']:np.testing.assert_array_equal(data[key][lo:lo+n],source[key][old:old+n])
 assert maxsum<=60.0001 and maxpoint<=30.0001 and maxslew<=150.001
 result=dict(clips=len(m['clips']),frames=len(cache),source_clip_checks=len(chosen),max_total_N=maxsum,max_point_N=maxpoint,max_slew_N_s=maxslew,source_copy_exact=True,index_alignment=True,finite=True)
 (a.cache/'validation.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
