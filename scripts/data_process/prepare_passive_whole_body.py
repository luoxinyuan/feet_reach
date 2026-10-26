"""Read-only views of complete local LIMMT clips; never crop motion windows."""
import json,hashlib
from pathlib import Path
import numpy as np
source=Path('/home/xl521/hiearchical_compliance/dataset/limmt_no_foot_gentle_amass_full')
root=Path(__file__).resolve().parents[2];out=root/'dataset/passive_whole_body_full'
meta=json.loads((source/'meta_motion.json').read_text());manifest={'source':str(source),'fps_assumed':50,'split':{},'provenance':'Split by stored clip ID; original recording identities unavailable, so recording-level independence is not guaranteed.'}
for split in ['train','val']:
 ids=[i for i in range(len(meta['starts'])) if (int(hashlib.sha256(str(i).encode()).hexdigest()[:8],16)%20==0)==(split=='val')]
 dest=out/split;dest.mkdir(parents=True,exist_ok=True)
 for name in ['meta.json','_tensordict']:
  p=dest/name
  if not p.exists():p.symlink_to(source/name,target_is_directory=(source/name).is_dir())
 selected={**meta,'starts':[meta['starts'][i] for i in ids],'ends':[meta['ends'][i] for i in ids],'info':{}}
 (dest/'meta_motion.json').write_text(json.dumps(selected))
 lengths=np.array(selected['ends'])-selected['starts'];manifest['split'][split]={'clips':len(ids),'frames':int(lengths.sum()),'length_min':int(lengths.min()),'length_max':int(lengths.max()),'source_ids':ids}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print({s:{k:v for k,v in d.items() if k!='source_ids'} for s,d in manifest['split'].items()})
