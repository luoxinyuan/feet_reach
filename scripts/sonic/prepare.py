"""Fetch pinned official SONIC source/weights; extract tensor-only frozen actor.

Run with the training Python. Dependencies go into the ignored cache, leaving
Isaac Lab's torch/numpy installation unchanged.
"""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/sonic'
SOURCE_COMMIT = '45c03facbf848e0c743091b7d9c17952096ff908'
# Original release checkpoint revision (official Hugging Face history).
HF_REVISION = 'bfd764a'
METADATA_GLOBALS = {
    'trl.trainer.ppo_config.PPOConfig', 'trl.trainer.utils.OnlineTrainerState',
    'transformers.trainer_utils.HubStrategy', 'transformers.trainer_utils.SchedulerType',
    'transformers.trainer_utils.IntervalStrategy', 'transformers.trainer_utils.SaveStrategy',
    'transformers.training_args.OptimizerNames', 'transformers.trainer_pt_utils.AcceleratorConfig',
    'accelerate.state.PartialState', 'accelerate.utils.dataclasses.DistributedType',
}

class IgnoredMetadata:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

def deque_as_list(*args):
    return list(args[0] if args else [])

def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-onnx', action='store_true', help='Also fetch official exports for parity validation.')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / 'source'
    if not source.exists():
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/NVlabs/GR00T-WholeBodyControl.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f'SONIC source must be at {SOURCE_COMMIT}; found {actual}')
    for name in ('config.yaml', 'last.pt'):
        path = CACHE / name
        # Always use the pinned version; a receipt identifies completed downloads.
        receipt = CACHE / (name + '.revision')
        if not receipt.exists() or receipt.read_text() != HF_REVISION:
            tmp = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(f'https://huggingface.co/nvidia/GEAR-SONIC/resolve/{HF_REVISION}/sonic_release/{name}', tmp)
            tmp.replace(path)
            receipt.write_text(HF_REVISION)
    if args.with_onnx:
        for name in ('model_encoder.onnx', 'model_decoder.onnx'):
            tmp = CACHE / (name + '.part')
            urllib.request.urlretrieve(f'https://huggingface.co/nvidia/GEAR-SONIC/resolve/{HF_REVISION}/{name}', tmp)
            tmp.replace(CACHE / name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(CACHE / 'deps'), '--upgrade', '--no-deps',
                    'loguru==0.7.3', 'vector-quantize-pytorch==1.27.15', 'einx==0.4.3', 'frozendict==2.4.7'], check=True)
    import torch
    found = set(torch.serialization.get_unsafe_globals_in_checkpoint(CACHE / 'last.pt'))
    if found - METADATA_GLOBALS - {'collections.deque'}:
        raise RuntimeError(f'Unexpected pickle globals: {found}')
    # Discard optimizer/trainer metadata without importing or executing trainer classes.
    allowed = [(IgnoredMetadata, name) for name in METADATA_GLOBALS]
    allowed.append((deque_as_list, 'collections.deque'))
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(CACHE / 'last.pt', map_location='cpu', weights_only=True)
    state = checkpoint.get('actor_model_state_dict', checkpoint.get('policy_state_dict'))
    assert state and all(isinstance(v, torch.Tensor) for v in state.values())
    torch.save(state, CACHE / 'actor.pt')
    manifest = {'source_commit': SOURCE_COMMIT, 'hf_revision': HF_REVISION,
                'variant': 'sonic_release', 'files': {n: sha256(CACHE / n) for n in ('config.yaml', 'last.pt', 'actor.pt')}}
    (CACHE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
