import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.foot_reach.checkpoints import parse_run, resolve_checkpoint, select_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_urls(self):
        run = 'luoxinyuan-duke-university/wall-foot-reach/wall-foot-reach-finetune-20261004_164441'
        entity, project, identifier = run.split('/')
        for prefix in ('https://forge.coreweave.com/wandb', 'https://wandb.ai'):
            self.assertEqual(parse_run(f'{prefix}/{entity}/{project}/runs/{identifier}?nw=user'), run)
        self.assertEqual(parse_run('run:' + run), run)
        self.assertIsNone(parse_run('/tmp/checkpoint_final.pt'))
        for source in ('run:../project/id', 'https://example.com/e/p/runs/id', 'run:e/p'):
            with self.assertRaises(ValueError):
                parse_run(source)

    def test_selection(self):
        files = [types.SimpleNamespace(name=n) for n in
                 ('checkpoint_900000.pt', 'checkpoint_final.pt', 'checkpoint_12.pt', 'checkpoint_bad.pt')]
        self.assertEqual(select_checkpoint(files).name, 'checkpoint_final.pt')
        self.assertEqual(select_checkpoint([files[0], files[2]]).name, 'checkpoint_900000.pt')
        self.assertEqual(select_checkpoint(files, 'checkpoint_12.pt').name, 'checkpoint_12.pt')
        for candidates in ([], [files[1], files[1]]):
            with self.assertRaises(ValueError):
                select_checkpoint(candidates)

    def test_download_cache_update_and_failure(self):
        class File:
            name = 'files/checkpoint_final.pt'
            md5 = 'revision1'
            size = 4
            calls = 0
            fail = False

            def download(self, root, replace):
                self.calls += 1
                path = Path(root) / self.name
                path.parent.mkdir(parents=True)
                path.write_bytes(b'test')
                if self.fail:
                    raise OSError('interrupted')
                return path.open()

        file = File()
        api = types.SimpleNamespace(settings={'base_url': 'https://api.wandb.ai'},
                                    run=lambda _: types.SimpleNamespace(files=lambda: [file]))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'wandb': types.SimpleNamespace(Api=lambda **_: api)}):
            first = resolve_checkpoint('run:e/p/id', directory)
            self.assertEqual(first.read_bytes(), b'test')
            self.assertEqual(resolve_checkpoint('run:e/p/id', directory), first)
            self.assertEqual(file.calls, 1)
            file.md5 = 'revision2'
            self.assertNotEqual(resolve_checkpoint('run:e/p/id', directory), first)
            self.assertEqual(file.calls, 2)
            file.md5 = 'revision3'
            file.fail = True
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                resolve_checkpoint('run:e/p/id', directory)
            self.assertEqual(len(list(Path(directory).rglob('*.pt'))), 2)
            file.fail = False
            file.name = '../checkpoint_final.pt'
            with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                resolve_checkpoint('run:e/p/id', directory)

    def test_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint_final.pt'
            path.touch()
            self.assertEqual(resolve_checkpoint(str(path), directory), path.resolve())
            with self.assertRaises(ValueError):
                resolve_checkpoint(str(path), directory, 'checkpoint_final.pt')


if __name__ == '__main__':
    unittest.main()
