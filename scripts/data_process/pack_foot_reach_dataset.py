"""Convert validated generated clips to the exact existing MotionData memmap."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from active_adaptation.utils.motion import MotionDataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/foot_reach')
    p.add_argument('--output',type=Path,default=ROOT/'dataset/wall_foot_reach')
    args=p.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text())
    if manifest['fps'] != 50: raise ValueError('Training task uses 50 Hz')
    for split in ('train','val'):
        source=args.source/'motions'/split
        if not source.exists(): continue
        destination=args.output/split
        if destination.exists(): raise FileExistsError(destination)
        # Deliberately no preprocess_motion: absolute support/wall coordinates
        # and the sole-to-ankle offset were already handled by the generator.
        source_files=[]
        def record_source(context, motion):
            source_files.append(str(Path(context["p"]).relative_to(args.source)))
        MotionDataset.create_from_path(str(source),target_fps=50,mem_path=str(destination),
            pad_before=0,pad_after=0,segment_len=1000,storage_float_dtype=torch.float32,
            storage_int_dtype=torch.int32,callback=record_source)
        (destination/"source_files.json").write_text(json.dumps(source_files,indent=2)+"\n")
        ds=MotionDataset.create_from_path_lazy(str(destination.resolve()))
        assert ds.joint_names == manifest['joint_names']
        assert ds.body_names == manifest['body_names']
        assert bool((ds.lengths>50).all())
        print(f'{split}: {len(ds.starts)} clips, {ds.data.joint_pos.shape[0]} frames, '
              f'{len(ds.joint_names)} joints, {len(ds.body_names)} bodies')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
