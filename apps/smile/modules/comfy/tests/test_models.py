"""Tests for the model tool (modules.comfy.models): the manifest, the checks, the installs from a mirror,
from a file in the repo and from the network (a fake http keyed by URL), and the exit codes."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.comfy import models
from modules.comfy.models import Entry, Manifest, Response


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHttp:
    """Routes keyed by URL; every call is recorded as (method, url, headers)."""

    def __init__(self) -> None:
        self.routes: dict = {}
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, url: str, method: str = 'GET', headers: dict | None = None) -> Response:
        self.calls.append((method, url, dict(headers or {})))
        if url not in self.routes:
            raise OSError(f'no route for {url}')
        return self.routes[url](method, dict(headers or {}))

    def lfs(self, url: str, cdn: str, data: bytes, size: int | None = None) -> None:
        """A Hugging Face resolve URL: 302 to the CDN with the LFS hash and size."""
        def handler(method: str, headers: dict) -> Response:
            return Response(302, {'location': cdn, 'x-linked-etag': f'"{sha(data)}"', 'x-linked-size': str(len(data) if size is None else size)})
        self.routes[url] = handler

    def cdn(self, url: str, data: bytes) -> None:
        """A CDN URL: honours Range."""
        def handler(method: str, headers: dict) -> Response:
            if 'Range' in headers:
                offset = int(headers['Range'].split('=')[1].rstrip('-'))
                return Response(206, {}, io.BytesIO(data[offset:]))
            return Response(200, {}, io.BytesIO(data))
        self.routes[url] = handler

    def plain(self, url: str, data: bytes) -> None:
        """A small file served directly (no LFS): 200 with a git etag and the length."""
        def handler(method: str, headers: dict) -> Response:
            return Response(200, {'etag': '"abc123"', 'content-length': str(len(data))}, io.BytesIO(data) if method == 'GET' else None)
        self.routes[url] = handler

    def gets(self, url: str) -> list[dict]:
        return [headers for method, called, headers in self.calls if method == 'GET' and called == url]


class ModelsCase(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)                 # stands in for the repo root (the home of repo-file sources)
        self.models_dir = self.root / 'models'
        self.models_dir.mkdir()
        self.http = FakeHttp()
        self.addCleanup(setattr, models, 'http', models.http)
        models.http = self.http
        quiet = mock.patch('sys.stdout', new_callable=io.StringIO)     # the commands print their summaries
        self.stdout = quiet.start()
        self.addCleanup(quiet.stop)
        self.data_a = b'model A ' * 100
        self.data_b = b'model B ' * 50
        self.manifest = Manifest([
            Entry('checkpoints/a.safetensors', len(self.data_a), sha(self.data_a), 'https://huggingface.co/pub/a/resolve/main/a.safetensors', 'public'),
            Entry('loras/b.safetensors', len(self.data_b), sha(self.data_b), 'data/comfy/loras/b.safetensors', 'in the repo'),
        ], self.root / 'models.json')
        self.manifest.save()

    def put(self, folder: Path, path: str, data: bytes) -> Path:
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def route_public(self, entry: Entry, data: bytes) -> str:
        cdn = f'https://cdn.example/{entry.path}'
        self.http.lfs(entry.source, cdn, data)
        self.http.cdn(cdn, data)
        return cdn


class TestManifest(ModelsCase):
    def test_round_trip(self) -> None:
        loaded = Manifest.load(self.manifest.path)
        self.assertEqual([e.to_dict() for e in loaded.files], [e.to_dict() for e in self.manifest.files])
        self.assertTrue(loaded.files[0].source_is_url)
        self.assertFalse(loaded.files[1].source_is_url)

    def test_hash_write_keeps_source_and_note_and_adds(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', b'changed')
        self.put(self.models_dir, 'loras/new.safetensors', b'new one')
        models.hash_files(self.manifest, self.models_dir, write=True, add=['loras\\new.safetensors'])
        self.assertIn('CHANGED', self.stdout.getvalue())
        loaded = Manifest.load(self.manifest.path)
        a = loaded.get('checkpoints/a.safetensors')
        self.assertEqual((a.size, a.sha256, a.source, a.note), (7, sha(b'changed'), 'https://huggingface.co/pub/a/resolve/main/a.safetensors', 'public'))
        new = loaded.get('loras/new.safetensors')
        self.assertEqual((new.size, new.sha256, new.source), (7, sha(b'new one'), None))
        self.assertEqual(loaded.get('loras/b.safetensors').sha256, sha(self.data_b))      # absent: untouched


class TestCheck(ModelsCase):
    def test_states_and_extras(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', b'x' * len(self.data_b))      # right size, wrong bytes
        self.put(self.models_dir, 'loras/put_loras_here', b'')
        self.put(self.models_dir, 'loras/.cache/huggingface/x.metadata', b'meta')
        self.put(self.models_dir, 'loras/other.safetensors', b'extra')
        self.put(self.models_dir, 'loras/half.safetensors.part', b'half')
        code = models.check(self.manifest, self.models_dir)
        lines = self.stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn('ok          checkpoints/a.safetensors', lines)
        self.assertIn('corrupt     loras/b.safetensors', lines)
        self.assertIn('extra       loras/other.safetensors', lines)
        self.assertNotIn('put_loras_here', lines)
        self.assertNotIn('.cache', lines)
        self.assertNotIn('.part', lines)
        self.assertEqual(models.status(self.manifest.files[1], self.models_dir, quick=True), 'ok')     # quick stops at the size
        (self.models_dir / 'loras' / 'b.safetensors').write_bytes(b'short')
        self.assertEqual(models.status(self.manifest.files[1], self.models_dir), 'wrong size')
        self.assertEqual(models.status(Entry('loras/none.safetensors', 1, 'x'), self.models_dir), 'missing')
        self.assertEqual(models.status(Entry('checkpoints/a.safetensors', len(self.data_a), ''), self.models_dir), 'no hash')


class TestInstall(ModelsCase):
    def test_mirror_first_then_the_sources(self) -> None:
        mirror = self.root / 'mirror'
        self.put(mirror, 'checkpoints/a.safetensors', self.data_a)
        self.put(mirror, 'loras/b.safetensors', b'y' * len(self.data_b))      # a bad mirror copy
        self.put(self.root, 'data/comfy/loras/b.safetensors', self.data_b)    # the repo has the right one
        code = models.install(self.manifest, self.models_dir, mirror, root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual((self.models_dir / 'checkpoints' / 'a.safetensors').read_bytes(), self.data_a)
        self.assertEqual((self.models_dir / 'loras' / 'b.safetensors').read_bytes(), self.data_b)
        self.assertEqual(self.http.calls, [])                                   # nothing went to the network
        self.assertFalse(list(self.models_dir.rglob('*.part')))

    def test_public_source_downloaded_and_verified(self) -> None:
        entry = self.manifest.files[0]
        cdn = self.route_public(entry, self.data_a)
        code = models.install(Manifest([entry], self.manifest.path), self.models_dir, root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual((self.models_dir / entry.path).read_bytes(), self.data_a)
        self.assertEqual(len(self.http.gets(cdn)), 1)

    def test_repo_file_missing_is_reported(self) -> None:
        code = models.install(Manifest([self.manifest.files[1]], self.manifest.path), self.models_dir, root=self.root)
        self.assertEqual(code, 1)
        self.assertIn('loras/b.safetensors', self.stdout.getvalue())

    def test_source_that_holds_another_file_is_skipped_before_download(self) -> None:
        entry = self.manifest.files[0]
        self.http.lfs(entry.source, 'https://cdn.example/other', b'something else')
        self.assertFalse(models.download(entry, entry.source, self.models_dir / entry.path))
        self.assertEqual(self.http.gets('https://cdn.example/other'), [])

    def test_resume_sends_range_and_leaves_no_part(self) -> None:
        entry = self.manifest.files[0]
        cdn = self.route_public(entry, self.data_a)
        part = self.put(self.models_dir, entry.path + '.part', self.data_a[:100])
        self.assertTrue(models.download(entry, entry.source, self.models_dir / entry.path))
        self.assertEqual(self.http.gets(cdn)[0].get('Range'), 'bytes=100-')
        self.assertEqual((self.models_dir / entry.path).read_bytes(), self.data_a)
        self.assertFalse(part.exists())

    def test_wrong_bytes_delete_the_part(self) -> None:
        entry = self.manifest.files[0]
        self.http.lfs(entry.source, 'https://cdn.example/bad', self.data_a)      # announces the right hash
        self.http.cdn('https://cdn.example/bad', b'z' * len(self.data_a))        # delivers other bytes
        self.assertFalse(models.download(entry, entry.source, self.models_dir / entry.path))
        self.assertFalse([p for p in self.models_dir.rglob('*') if p.is_file()])      # neither the file nor a .part

    def test_entry_without_hash_is_not_installed(self) -> None:
        entry = Entry('loras/x.safetensors', 3, '', None)
        self.put(self.root / 'mirror', entry.path, b'abc')
        code = models.install(Manifest([entry], self.manifest.path), self.models_dir, self.root / 'mirror', root=self.root)
        self.assertEqual(code, 1)
        self.assertFalse((self.models_dir / entry.path).exists())

    def test_nothing_to_do_when_all_ok(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
        self.assertEqual(models.install(self.manifest, self.models_dir, root=self.root), 0)
        self.assertEqual(self.http.calls, [])


class TestSources(ModelsCase):
    def test_public_adopt_ok_mismatch_plain_and_repo_files(self) -> None:
        fresh = Entry('geometry/d.safetensors', 0, '', 'https://huggingface.co/pub/d/resolve/main/d.safetensors')
        plain = Entry('llm/config.json', 5, 'whatever', 'https://huggingface.co/pub/q/resolve/main/config.json')
        wrong = Entry('loras/w.safetensors', 4, sha(b'mine'), 'https://huggingface.co/pub/w/resolve/main/w.safetensors')
        gone = Entry('loras/g.safetensors', 2, sha(b'gg'), 'data/comfy/loras/g.safetensors')
        manifest = Manifest([self.manifest.files[0], fresh, plain, wrong, self.manifest.files[1], gone], self.manifest.path)
        self.http.lfs(self.manifest.files[0].source, 'https://cdn.example/a', self.data_a)
        self.http.lfs(fresh.source, 'https://cdn.example/d', b'depth' * 10)
        self.http.plain(plain.source, b'{...}')
        self.http.lfs(wrong.source, 'https://cdn.example/w', b'ours')
        self.put(self.root, 'data/comfy/loras/b.safetensors', self.data_b)
        code = models.sources(manifest, write=True, root=self.root)
        lines = self.stdout.getvalue()
        self.assertEqual(code, 1)
        self.assertIn('ok                   checkpoints/a.safetensors', lines)
        self.assertIn('adopted              geometry/d.safetensors', lines)
        self.assertIn('size ok              llm/config.json', lines)
        self.assertIn('MISMATCH             loras/w.safetensors', lines)
        self.assertIn('ok                   loras/b.safetensors', lines)
        self.assertIn('missing in repo      loras/g.safetensors', lines)
        loaded = Manifest.load(self.manifest.path)
        self.assertEqual((loaded.get('geometry/d.safetensors').sha256, loaded.get('geometry/d.safetensors').size), (sha(b'depth' * 10), 50))
        self.assertEqual(loaded.get('loras/w.safetensors').source, wrong.source)        # kept without --drop-mismatch
        self.assertEqual(len([c for c in self.http.calls if c[0] == 'GET']), 0)       # nothing downloaded
        models.sources(Manifest.load(self.manifest.path), write=True, drop_mismatch=True, root=self.root)
        self.assertIsNone(Manifest.load(self.manifest.path).get('loras/w.safetensors').source)


class TestMain(ModelsCase):
    def test_exit_codes(self) -> None:
        args = ['--manifest', str(self.manifest.path), '--models', str(self.models_dir)]
        self.assertEqual(models.main(['check'] + args), 1)
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
        self.assertEqual(models.main(['check'] + args), 0)
        self.assertEqual(models.main(['check', '--quick'] + args), 0)
        self.assertEqual(models.main(['check', '--manifest', str(self.root / 'none.json')]), 1)


if __name__ == '__main__':
    unittest.main()
