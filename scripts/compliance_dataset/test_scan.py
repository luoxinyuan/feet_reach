"""Small regression checks, including the manifest-writing CLI path."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scan import inspect, windows, merge

q=np.zeros((200,36),dtype=np.float32);q[:,2]=.75;q[:,3]=1
assert inspect(q,50)[0]==[] and 'still' in inspect(q,50)[1]
bad=q.copy();bad[100:,7]=1
assert 'joint_jump' in inspect(bad,50)[0]
flip=q.copy();flip[50:100,3]=-1
assert not inspect(flip,50)[0]
bad_nan=q.copy();bad_nan[3,0]=np.nan
assert inspect(bad_nan,50)[0]==['nonfinite']
assert windows(49,50)==[] and windows(50,50)==[(0,50)]
assert windows(250,50)==[(0,200),(50,250)]
assert merge([(0,200),(100,300),(400,500)])==[[0,300],[400,500]]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);src=root/'source.npz'
 np.savez(src,qpos=np.concatenate([q,bad]),starts=[0,200],lengths=[200,200],names=['good','jump'],fps=50.)
 original=src.read_bytes()
 subprocess.run([sys.executable,str(Path(__file__).with_name('scan.py')),'--datasets','lafan','--lafan',str(src),'--output',str(root/'out')],check=True)
 result=json.loads((root/'out/lafan.allowlist.json').read_text())
 assert [c['clip_id'] for c in result['clips']]==[0]
 assert result['summary']['allowed_union_hours']==4/3600
 assert src.read_bytes()==original
print('PASS: quality gates, quaternion signs, window coverage, CLI allowlist, source unchanged')
