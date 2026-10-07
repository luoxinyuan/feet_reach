"""Resolve local checkpoints or download checkpoint files from a W&B run."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import unquote, urlsplit


def parse_run(source):
    """Return entity/project/run-id, or None for a local filename."""
    if source.startswith(('https://', 'http://')):
        url = urlsplit(source)
        parts = unquote(url.path).strip('/').split('/')
        if url.hostname == 'forge.coreweave.com' and parts[0] == 'wandb':
            parts = parts[1:]
        elif url.hostname not in ('wandb.ai', 'www.wandb.ai'):
            raise ValueError('Use a W&B/Forge run URL, or run:entity/project/run-id with WANDB_BASE_URL for a private server')
        if len(parts) < 4 or parts[2] != 'runs':
            raise ValueError('Expected a W&B URL containing /entity/project/runs/run-id')
        parts = [parts[0], parts[1], parts[3]]
    elif source.startswith('run:'):
        parts = source[4:].split('/')
    else:
        return None
    if len(parts) != 3 or any(not re.fullmatch(r'[\w.-]+', p) or p in ('.', '..') for p in parts):
        raise ValueError('Expected run:entity/project/run-id')
    return '/'.join(parts)


def select_checkpoint(files, filename=None):
    files = list(files)
    if filename:
        matches = [f for f in files if f.name == filename]
    else:
        ranked = []
        for f in files:
            match = re.fullmatch(r'checkpoint_(final|\d+)\.(pt|ckpt|pth)', PurePosixPath(f.name).name)
            if match:
                tag = match[1]
                ranked.append(((tag == 'final', int(tag) if tag != 'final' else 0), f))
        best = max((rank for rank, _ in ranked), default=None)
        matches = [f for rank, f in ranked if rank == best]
    if not matches:
        raise ValueError('No matching checkpoint in W&B run Files; check the run finished uploading or use --wandb-file')
    if len(matches) != 1:
        raise ValueError('Multiple matching checkpoints; specify the exact run Files path with --wandb-file')
    return matches[0]


def resolve_checkpoint(source, cache_dir, filename=None):
    run_path = parse_run(source)
    if run_path is None:
        if filename:
            raise ValueError('--wandb-file requires a W&B run URL or run:entity/project/run-id')
        return Path(source).expanduser().resolve(strict=True)
    try:
        import wandb
    except ImportError as exc:
        raise ValueError('W&B download requires wandb in the evaluation Python environment: python -m pip install wandb') from exc
    # The Forge URL is a UI link. Let the SDK use its configured API endpoint
    # (api.wandb.ai by default), including WANDB_BASE_URL for private servers.
    api = wandb.Api(timeout=30)
    try:
        checkpoint = select_checkpoint(api.run(run_path).files(), filename)
        remote = PurePosixPath(checkpoint.name)
        if remote.is_absolute() or '..' in remote.parts or '\\' in checkpoint.name:
            raise ValueError('Invalid checkpoint file path returned by W&B')
        identity = f'{api.settings.get("base_url")}:{run_path}:{checkpoint.name}:{checkpoint.md5}'
        cache = Path(cache_dir) / hashlib.sha256(identity.encode()).hexdigest()
        target = cache / remote
        if target.is_file() and target.stat().st_size == checkpoint.size:
            print(f'Using cached W&B checkpoint: {target}', flush=True)
            return target.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        print(f'Downloading W&B checkpoint: {run_path}/{checkpoint.name}', flush=True)
        # A failed download never becomes a usable cache entry.
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            checkpoint.download(root=temporary, replace=True).close()
            downloaded = Path(temporary) / remote
            if downloaded.stat().st_size != checkpoint.size:
                raise ValueError('Downloaded checkpoint size does not match W&B metadata')
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
        return target.resolve(strict=True)
    except Exception as exc:
        raise ValueError(f'Cannot download checkpoint from {run_path}: {exc}. Check wandb login / WANDB_API_KEY and run access.') from exc
