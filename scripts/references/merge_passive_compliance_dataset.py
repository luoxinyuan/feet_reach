"""Combine separately validated left/right arm data, preserving endpoint splits."""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'artifacts/passive_compliance'
if (out/'manifest.json').exists(): raise FileExistsError(out/'manifest.json')
manifests=[json.loads((out/side/'manifest.json').read_text()) for side in ('left','right')]
manifest={k:v for k,v in manifests[0].items() if k not in ('clips','checks','rejections')}
manifest['clips']=[]
manifest['checks_by_side']={s:m['checks'] for s,m in zip(('left','right'),manifests)}
for side,m in zip(('left','right'),manifests):
    assert m['joint_names']==manifest['joint_names'] and m['body_names']==manifest['body_names']
    for clip in m['clips']:
        dst=out/clip['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/side/clip['path'],dst)
        manifest['clips'].append(dict(clip,side=side))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('MERGED',len(manifest['clips']),sum(c['frames'] for c in manifest['clips']))
