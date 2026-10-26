"""Tests for the model tool (modules.comfy.models): the manifest, the checks, the installs from a mirror
and from the network (a fake http keyed by URL), the token lookup, export and the exit codes."""

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.comfy import models
from modules.comfy.models import Entry, Manifest, Response

HF_REPO = 'someone/smile-models'


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
        """A CDN URL: refuses a token, honours Range."""
        def handler(method: str, headers: dict) -> Response:
            if 'Authorization' in headers:
                return Response(401, {})
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

    def heads(self, url: str) -> list[dict]:
        return [headers for method, called, headers in self.calls if method == 'HEAD' and called == url]


class ModelsCase(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models_dir = self.root / 'models'
        self.models_dir.mkdir()
        self.http = FakeHttp()
        self.addCleanup(setattr, models, 'http', models.http)
        models.http = self.http
        quiet = mock.patch('sys.stdout', new_callable=io.StringIO)     # the commands print their summaries
        quiet.start()
        self.addCleanup(quiet.stop)
        self.data_a = b'model A ' * 100
        self.data_b = b'model B ' * 50
        self.manifest = Manifest(HF_REPO, [
            Entry('checkpoints/a.safetensors', len(self.data_a), sha(self.data_a), 'https://huggingface.co/pub/a/resolve/main/a.safetensors', 'a'),
            Entry('loras/b.safetensors', len(self.data_b), sha(self.data_b), None, 'private'),
        ], self.root / 'models.json')
        self.manifest.save()

    def put(self, folder: Path, path: str, data: bytes) -> Path:
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def route_hf(self, entry: Entry, data: bytes) -> None:
        cdn = f'https://cdn.example/{entry.path}'
        self.http.lfs(self.manifest.hf_url(entry), cdn, data)
        self.http.cdn(cdn, data)


class TestManifest(ModelsCase):
    def test_round_trip_and_urls(self) -> None:
        loaded = Manifest.load(self.manifest.path)
        self.assertEqual([e.to_dict() for e in loaded.files], [e.to_dict() for e in self.manifest.files])
        self.assertEqual(loaded.repo, HF_REPO)
        self.assertTrue(loaded.repo_ready)
        self.assertEqual(loaded.hf_url(loaded.files[1]), f'https://huggingface.co/{HF_REPO}/resolve/main/loras/b.safetensors')
        self.assertFalse(Manifest('<hf-user>/smile-models').repo_ready)
        self.assertFalse(Manifest('').repo_ready)

    def test_hash_write_keeps_source_and_note_and_adds(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', b'changed')
        self.put(self.models_dir, 'loras/new.safetensors', b'new one')
        with mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            models.hash_files(self.manifest, self.models_dir, write=True, add=['loras\\new.safetensors'])
        self.assertIn('CHANGED', out.getvalue())
        self.assertIn('new', out.getvalue())
        loaded = Manifest.load(self.manifest.path)
        a = loaded.get('checkpoints/a.safetensors')
        self.assertEqual((a.size, a.sha256, a.source, a.note), (7, sha(b'changed'), 'https://huggingface.co/pub/a/resolve/main/a.safetensors', 'a'))
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
        with mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            code = models.check(self.manifest, self.models_dir)
        lines = out.getvalue()
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
    def test_mirror_copy_verified_and_corrupt_copy_rejected(self) -> None:
        mirror = self.root / 'mirror'
        self.put(mirror, 'checkpoints/a.safetensors', self.data_a)
        self.put(mirror, 'loras/b.safetensors', b'y' * len(self.data_b))      # a bad mirror copy
        self.route_hf(self.manifest.files[1], self.data_b)                   # the HF repo has the right one
        code = models.install(self.manifest, self.models_dir, mirror, token_getter=lambda: 'tok')
        self.assertEqual(code, 0)
        self.assertEqual((self.models_dir / 'checkpoints' / 'a.safetensors').read_bytes(), self.data_a)
        self.assertEqual((self.models_dir / 'loras' / 'b.safetensors').read_bytes(), self.data_b)
        self.assertEqual(self.http.gets('https://huggingface.co/pub/a/resolve/main/a.safetensors'), [])     # a came from the mirror
        self.assertFalse(list(self.models_dir.rglob('*.part')))

    def test_token_only_on_the_hf_head_never_on_the_cdn(self) -> None:
        self.route_hf(self.manifest.files[1], self.data_b)
        code = models.install(self.manifest.__class__(HF_REPO, [self.manifest.files[1]], self.manifest.path), self.models_dir, token_getter=lambda: 'tok')
        self.assertEqual(code, 0)
        head = self.http.heads(self.manifest.hf_url(self.manifest.files[1]))
        self.assertEqual(head[0].get('Authorization'), 'Bearer tok')
        get = self.http.gets('https://cdn.example/loras/b.safetensors')
        self.assertEqual(len(get), 1)
        self.assertNotIn('Authorization', get[0])

    def test_no_token_skips_the_repo_and_takes_the_source(self) -> None:
        entry = self.manifest.files[0]
        cdn = 'https://cdn.example/pub/a'
        self.http.lfs(entry.source, cdn, self.data_a)
        self.http.cdn(cdn, self.data_a)
        code = models.install(Manifest(HF_REPO, [entry], self.manifest.path), self.models_dir, token_getter=lambda: None)
        self.assertEqual(code, 0)
        self.assertEqual(self.http.heads(self.manifest.hf_url(entry)), [])
        self.assertEqual((self.models_dir / entry.path).read_bytes(), self.data_a)

    def test_source_that_holds_another_file_is_skipped_before_download(self) -> None:
        entry = self.manifest.files[0]
        self.http.lfs(entry.source, 'https://cdn.example/other', b'something else')
        self.assertFalse(models.download(entry, entry.source, self.models_dir / entry.path))
        self.assertEqual(self.http.gets('https://cdn.example/other'), [])

    def test_resume_sends_range_and_leaves_no_part(self) -> None:
        entry = self.manifest.files[1]
        self.route_hf(entry, self.data_b)
        part = self.put(self.models_dir, entry.path + '.part', self.data_b[:100])
        self.assertTrue(models.download(entry, self.manifest.hf_url(entry), self.models_dir / entry.path, 'tok'))
        get = self.http.gets('https://cdn.example/loras/b.safetensors')
        self.assertEqual(get[0].get('Range'), 'bytes=100-')
        self.assertEqual((self.models_dir / entry.path).read_bytes(), self.data_b)
        self.assertFalse(part.exists())

    def test_wrong_bytes_delete_the_part_and_fall_to_the_source(self) -> None:
        entry = self.manifest.files[0]
        # the HF repo announces the right hash but delivers other bytes
        bad = b'z' * len(self.data_a)
        self.http.lfs(self.manifest.hf_url(entry), 'https://cdn.example/bad', self.data_a)
        self.http.cdn('https://cdn.example/bad', bad)
        self.http.lfs(entry.source, 'https://cdn.example/good', self.data_a)
        self.http.cdn('https://cdn.example/good', self.data_a)
        code = models.install(Manifest(HF_REPO, [entry], self.manifest.path), self.models_dir, token_getter=lambda: 'tok')
        self.assertEqual(code, 0)
        self.assertEqual((self.models_dir / entry.path).read_bytes(), self.data_a)
        self.assertFalse(list(self.models_dir.rglob('*.part')))

    def test_entry_without_hash_is_not_installed(self) -> None:
        entry = Entry('loras/x.safetensors', 3, '', None)
        self.put(self.root / 'mirror', entry.path, b'abc')
        code = models.install(Manifest(HF_REPO, [entry], self.manifest.path), self.models_dir, self.root / 'mirror', token_getter=lambda: None)
        self.assertEqual(code, 1)
        self.assertFalse((self.models_dir / entry.path).exists())

    def test_nothing_to_do_when_all_ok(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
        self.assertEqual(models.install(self.manifest, self.models_dir, token_getter=lambda: 'tok'), 0)
        self.assertEqual(self.http.calls, [])


class TestToken(ModelsCase):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(setattr, models, 'HF_TOKEN_FILE', models.HF_TOKEN_FILE)
        models.HF_TOKEN_FILE = self.root / 'hf' / 'token'

    def test_env_then_file_then_prompt_saved(self) -> None:
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'from-env'}):
            self.assertEqual(models.get_token(), 'from-env')
        with mock.patch.dict(os.environ, {'HF_TOKEN': ''}):
            self.assertIsNone(models.get_token(prompt=False))
            models.HF_TOKEN_FILE.parent.mkdir()
            models.HF_TOKEN_FILE.write_text('from-file\n')
            self.assertEqual(models.get_token(), 'from-file')
            models.HF_TOKEN_FILE.unlink()
            with mock.patch.object(models.getpass, 'getpass', return_value='typed'), \
                    mock.patch.object(models.sys, 'stdin') as stdin:
                stdin.isatty.return_value = True
                self.assertEqual(models.get_token(), 'typed')
            self.assertEqual(models.HF_TOKEN_FILE.read_text(), 'typed')


class TestSources(ModelsCase):
    def test_adopt_ok_mismatch_and_plain(self) -> None:
        fresh = Entry('geometry/d.safetensors', 0, '', 'https://huggingface.co/pub/d/resolve/main/d.safetensors')
        plain = Entry('llm/config.json', 5, 'whatever', 'https://huggingface.co/pub/q/resolve/main/config.json')
        wrong = Entry('loras/w.safetensors', 4, sha(b'mine'), 'https://huggingface.co/pub/w/resolve/main/w.safetensors')
        manifest = Manifest(HF_REPO, [self.manifest.files[0], fresh, plain, wrong, self.manifest.files[1]], self.manifest.path)
        self.http.lfs(self.manifest.files[0].source, 'https://cdn.example/a', self.data_a)
        self.http.lfs(fresh.source, 'https://cdn.example/d', b'depth' * 10)
        self.http.plain(plain.source, b'{...}')
        self.http.lfs(wrong.source, 'https://cdn.example/w', b'ours')
        with mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            code = models.sources(manifest, write=True)
        lines = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn('ok                   checkpoints/a.safetensors', lines)
        self.assertIn('adopted              geometry/d.safetensors', lines)
        self.assertIn('size ok              llm/config.json', lines)
        self.assertIn('MISMATCH             loras/w.safetensors', lines)
        loaded = Manifest.load(self.manifest.path)
        self.assertEqual((loaded.get('geometry/d.safetensors').sha256, loaded.get('geometry/d.safetensors').size), (sha(b'depth' * 10), 50))
        self.assertEqual(loaded.get('loras/w.safetensors').source, wrong.source)        # kept without --drop-mismatch
        self.assertEqual(len([c for c in self.http.calls if c[0] == 'GET']), 0)       # nothing downloaded
        models.sources(Manifest.load(self.manifest.path), write=True, drop_mismatch=True)
        self.assertIsNone(Manifest.load(self.manifest.path).get('loras/w.safetensors').source)

    def test_repo_tree(self) -> None:
        tree = [{'type': 'file', 'path': 'checkpoints/a.safetensors', 'size': len(self.data_a), 'lfs': {'oid': sha(self.data_a), 'size': len(self.data_a)}},
                {'type': 'file', 'path': 'loras/b.safetensors', 'size': 1, 'lfs': {'oid': 'deadbeef', 'size': 1}}]
        import json
        self.http.routes[f'https://huggingface.co/api/models/{HF_REPO}/tree/main?recursive=true'] = \
            lambda method, headers: Response(200, {}, io.BytesIO(json.dumps(tree).encode()))
        self.http.lfs(self.manifest.files[0].source, 'https://cdn.example/a', self.data_a)
        with mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            code = models.sources(self.manifest, repo=True, token_getter=lambda: 'tok')
        self.assertEqual(code, 1)
        self.assertIn(f'ok                   {HF_REPO}/checkpoints/a.safetensors', out.getvalue())
        self.assertIn(f'MISMATCH             {HF_REPO}/loras/b.safetensors', out.getvalue())


class TestExportAndUpload(ModelsCase):
    def test_export_copies_skips_and_writes_manifest(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
        dest = self.root / 'usb'
        self.put(dest, 'loras/b.safetensors', self.data_b)      # already there and right
        before = (dest / 'loras' / 'b.safetensors').stat().st_mtime_ns
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(models.export(self.manifest, self.models_dir, dest), 0)
        self.assertEqual((dest / 'checkpoints' / 'a.safetensors').read_bytes(), self.data_a)
        self.assertEqual((dest / 'loras' / 'b.safetensors').stat().st_mtime_ns, before)
        self.assertEqual(Manifest.load(dest / 'models.json').repo, HF_REPO)
        (self.models_dir / 'loras' / 'b.safetensors').write_bytes(b'bad')
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(models.export(self.manifest, self.models_dir, self.root / 'usb2'), 1)   # nothing exported unverified
        self.assertFalse((self.root / 'usb2').exists())

    def test_upload_command(self) -> None:
        self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
        self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
        exe = self.put(self.root, 'hf.exe', b'')
        runs: list[list[str]] = []

        def run(command, check):
            runs.append(command)
            return mock.Mock(returncode=0)

        with mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            code = models.upload(self.manifest, self.models_dir, exe, token_getter=lambda: 'tok', run=run)
        self.assertEqual(code, 0)
        command = runs[0]
        self.assertEqual(command[:3], [str(exe), 'upload', HF_REPO])
        self.assertIn('--private', command)
        self.assertEqual([command[i + 1] for i, arg in enumerate(command) if arg == '--include'], ['checkpoints/a.safetensors', 'loras/b.safetensors'])
        self.assertIn('--token', command)
        self.assertNotRegex(out.getvalue(), r'--token tok')       # the token itself is not printed
        self.assertIn('--token <token>', out.getvalue())
        self.assertEqual(models.upload(Manifest('<hf-user>/x', self.manifest.files, self.manifest.path), self.models_dir, exe, token_getter=lambda: 'tok', run=run), 1)
        self.assertEqual(len(runs), 1)


class TestMain(ModelsCase):
    def test_exit_codes(self) -> None:
        args = ['--manifest', str(self.manifest.path), '--models', str(self.models_dir)]
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(models.main(['check'] + args), 1)
            self.put(self.models_dir, 'checkpoints/a.safetensors', self.data_a)
            self.put(self.models_dir, 'loras/b.safetensors', self.data_b)
            self.assertEqual(models.main(['check'] + args), 0)
            self.assertEqual(models.main(['check', '--quick'] + args), 0)
            self.assertEqual(models.main(['check', '--manifest', str(self.root / 'none.json')]), 1)


if __name__ == '__main__':
    unittest.main()
