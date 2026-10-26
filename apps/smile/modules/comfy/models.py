"""The model tool: keeps a models folder equal to the manifest comfy/models.json.

The manifest lists every model file ComfyUI needs for SMILE (path relative to the models folder, size,
sha256, a public source URL or null) and the private Hugging Face repo that holds the verified copies.
Standard library only, so it runs before any venv exists, from the repo root:

    py -3.12 -m modules.comfy.models check [--quick]            every file ok / missing / wrong size / corrupt, extras listed
    py -3.12 -m modules.comfy.models install [--mirror DIR]     fetch what is not ok: mirror copy, else the HF repo, else the source
    py -3.12 -m modules.comfy.models export DIR                 verify, then copy the set into DIR (a USB SSD, iCloud Drive)
    py -3.12 -m modules.comfy.models hash [--write] [--add P]   size and sha256 of the files present, into the manifest
    py -3.12 -m modules.comfy.models sources [--write] [--repo] the public sources (and the HF repo) against the manifest
    py -3.12 -m modules.comfy.models upload                     the verified set into the HF repo (runs hf.exe)

Defaults: the manifest in comfy/, the models folder beside the repo (<repo>\\..\\smile-models); --manifest and
--models override. The HF token comes from HF_TOKEN, else ~/.cache/huggingface/token (where `hf auth login`
writes it), else a prompt that saves it there; without one the HF repo is skipped and only the sources are tried.

A Hugging Face `resolve` URL answers 302 to a CDN URL on another host that refuses the Authorization header,
and that 302 carries the file's LFS sha256 (`x-linked-etag`) and size (`x-linked-size`): the HEAD with the
token checks a source against the manifest before a byte is downloaded, the GET goes to the CDN without the
token. Downloads go to `<path>.part` and resume; a file is renamed into place only after its sha256 matches.
Every network call goes through `http()`, which the tests replace.
"""

import argparse
import getpass
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

logger = logging.getLogger('models')

ROOT: Path = Path(__file__).resolve().parents[2]
MANIFEST: Path = ROOT / 'comfy' / 'models.json'
MODELS: Path = ROOT.parent / 'smile-models'
HF_TOKEN_FILE: Path = Path.home() / '.cache' / 'huggingface' / 'token'
HF_EXE: Path = ROOT / 'ComfyUI' / '.venv' / 'Scripts' / 'hf.exe'
CHUNK: int = 1 << 22
TIMEOUT: float = 60.0
PART: str = '.part'


# MANIFEST

@dataclass
class Entry:
    path: str                       # relative to the models folder, with forward slashes
    size: int = 0
    sha256: str = ''                # empty: not known yet (hash or sources --write fill it)
    source: str | None = None       # public URL, the fallback behind the HF repo; None: none
    note: str = ''

    @classmethod
    def from_dict(cls, data: dict) -> 'Entry':
        return cls(data['path'], int(data.get('size') or 0), data.get('sha256') or '', data.get('source'), data.get('note') or '')

    def to_dict(self) -> dict:
        return {'path': self.path, 'size': self.size, 'sha256': self.sha256, 'source': self.source, 'note': self.note}


@dataclass
class Manifest:
    repo: str
    files: list[Entry] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> 'Manifest':
        with open(path, encoding='utf-8') as f:
            data: dict = json.load(f)
        return cls(data.get('repo') or '', [Entry.from_dict(d) for d in data.get('files', [])], Path(path))

    def save(self, path: Path | None = None) -> None:
        target: Path = path or self.path
        if target is None:
            raise ValueError('the manifest has no path')
        with open(target, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({'repo': self.repo, 'files': [e.to_dict() for e in self.files]}, f, indent=2)
            f.write('\n')

    @property
    def repo_ready(self) -> bool:
        """False while repo is the placeholder of a fresh manifest."""
        return bool(self.repo) and '<' not in self.repo and '/' in self.repo

    def hf_url(self, entry: Entry) -> str:
        return f'https://huggingface.co/{self.repo}/resolve/main/{entry.path}'

    def get(self, path: str) -> Entry | None:
        return next((e for e in self.files if e.path == path), None)


# NETWORK (one seam, replaced by the tests)

@dataclass
class Response:
    status: int
    headers: dict[str, str]         # lower-case names
    body: BinaryIO | None = None    # readable for a GET

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(url: str, method: str = 'GET', headers: dict[str, str] | None = None) -> Response:
    """One request, redirects not followed (a 3xx comes back as the response). Raises OSError when
    the host cannot be reached at all."""
    request = urllib.request.Request(url, method=method)
    for name, value in (headers or {}).items():
        request.add_unredirected_header(name, value)      # never forwarded to a redirect target
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        reply = opener.open(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:                     # every non-2xx, the 302 included
        return Response(e.code, {k.lower(): v for k, v in e.headers.items()}, e)
    return Response(reply.status, {k.lower(): v for k, v in reply.headers.items()}, reply)


http: Callable[..., Response] = _http


@dataclass
class Resolved:
    url: str                        # where the bytes are (the CDN after a 302, else the URL itself)
    sha256: str | None              # the server's LFS hash, when it gives one
    size: int | None
    direct: bool                    # True: served by the URL itself, so the token goes with the GET


def resolve(url: str, token: str | None = None) -> Resolved | None:
    """HEAD a URL: where it leads and what it says about the file. None when it is not there."""
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'} if token else {}
    try:
        reply: Response = http(url, 'HEAD', headers)
    except OSError as e:
        logger.warning('%s: %s', url, e)
        return None
    if reply.status in (301, 302, 303, 307, 308):
        location: str | None = reply.header('Location')
        if not location:
            return None
        if location.startswith('/'):
            location = url.split('/', 3)[0] + '//' + url.split('/', 3)[2] + location
        return Resolved(location, _sha_of(reply.header('x-linked-etag')), _int_or_none(reply.header('x-linked-size')), False)
    if reply.status == 200:
        return Resolved(url, _sha_of(reply.header('x-linked-etag')), _int_or_none(reply.header('content-length')), True)
    logger.warning('%s: HTTP %s', url, reply.status)
    return None


def _sha_of(etag: str | None) -> str | None:
    """An etag that is a sha256 (LFS files), else None (small files carry a git hash)."""
    if etag is None:
        return None
    value: str = etag.strip().strip('"')
    if value.startswith('W/'):
        value = value[2:].strip('"')
    return value.lower() if len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower()) else None


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


# FILES

def file_sha256(path: Path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def status(entry: Entry, folder: Path, quick: bool = False) -> str:
    """ok, missing, wrong size or corrupt; quick stops at the size."""
    target: Path = folder / entry.path
    if not target.is_file():
        return 'missing'
    if target.stat().st_size != entry.size:
        return 'wrong size'
    if quick:
        return 'ok'
    if not entry.sha256:
        return 'no hash'
    return 'ok' if file_sha256(target) == entry.sha256 else 'corrupt'


def _progress(done: int, total: int, label: str, last: list[int]) -> None:
    """One \\r line on a terminal, else a line per 10 %."""
    if total <= 0:
        return
    percent: int = min(100, done * 100 // total)
    if sys.stderr.isatty():
        sys.stderr.write(f'\r  {label}: {percent:3d} % of {total / 1e6:,.0f} MB')
        if percent == 100:
            sys.stderr.write('\n')
        sys.stderr.flush()
    elif percent // 10 > last[0]:
        last[0] = percent // 10
        logger.info('  %s: %d %%', label, percent)


def _write_verified(entry: Entry, target: Path, reader: BinaryIO, digest: Any, mode: str, done: int) -> bool:
    """Stream reader into <target>.part (mode 'wb' or 'ab') while hashing; rename into place when the
    hash matches, else delete the part and say why."""
    part: Path = target.with_name(target.name + PART)
    target.parent.mkdir(parents=True, exist_ok=True)
    last: list[int] = [done * 10 // entry.size if entry.size else 0]
    with open(part, mode) as out:
        while chunk := reader.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            _progress(done, entry.size, entry.path, last)
    size: int = part.stat().st_size
    found: str = digest.hexdigest()
    if size != entry.size or found != entry.sha256:
        part.unlink()
        logger.error('%s: got %d bytes with sha256 %s, the manifest says %d and %s', entry.path, size, found, entry.size, entry.sha256)
        return False
    if target.exists():
        target.unlink()
    part.rename(target)
    return True


def copy_verified(entry: Entry, source: Path, target: Path) -> bool:
    """A mirror copy, hashed while it is copied."""
    logger.info('%s: copying from %s', entry.path, source)
    with open(source, 'rb') as f:
        return _write_verified(entry, target, f, hashlib.sha256(), 'wb', 0)


def download(entry: Entry, url: str, target: Path, token: str | None = None) -> bool:
    """Fetch a URL into place, resuming a .part, verified; False when the URL is not there, says it
    holds another file, or delivered one."""
    found: Resolved | None = resolve(url, token)
    if found is None:
        return False
    if found.sha256 is not None and found.sha256 != entry.sha256:
        logger.warning('%s: %s holds another file (sha256 %s), skipped', entry.path, url, found.sha256)
        return False
    if found.size is not None and found.size != entry.size:
        logger.warning('%s: %s holds %d bytes, the manifest says %d, skipped', entry.path, url, found.size, entry.size)
        return False
    part: Path = target.with_name(target.name + PART)
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'} if token and found.direct else {}
    digest = hashlib.sha256()
    offset: int = 0
    if part.is_file():
        with open(part, 'rb') as f:
            while chunk := f.read(CHUNK):
                digest.update(chunk)
                offset += len(chunk)
        headers['Range'] = f'bytes={offset}-'
    try:
        reply: Response = http(found.url, 'GET', headers)
    except OSError as e:
        logger.warning('%s: %s', url, e)
        return False
    if offset and reply.status == 206:
        logger.info('%s: resuming at %d bytes from %s', entry.path, offset, url)
        mode: str = 'ab'
    elif reply.status == 200:
        logger.info('%s: downloading from %s', entry.path, url)
        digest, offset, mode = hashlib.sha256(), 0, 'wb'
    else:
        logger.warning('%s: HTTP %s from %s', entry.path, reply.status, found.url)
        return False
    if reply.body is None:
        return False
    return _write_verified(entry, target, reply.body, digest, mode, offset)


# TOKEN

def get_token(prompt: bool = True) -> str | None:
    """HF_TOKEN, else the saved token, else one asked for once and saved."""
    token: str = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    if HF_TOKEN_FILE.is_file():
        token = HF_TOKEN_FILE.read_text(encoding='utf-8').strip()
        if token:
            return token
    if not prompt or not sys.stdin or not sys.stdin.isatty():
        return None
    token = getpass.getpass('Hugging Face token (write access to the models repo; saved to %s): ' % HF_TOKEN_FILE).strip()
    if not token:
        return None
    HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    HF_TOKEN_FILE.write_text(token, encoding='utf-8')
    return token


# COMMANDS

def _extras(manifest: Manifest, folder: Path) -> list[str]:
    """Files under the manifest's top-level folders that the manifest does not name."""
    known: set[str] = {e.path for e in manifest.files}
    tops: set[str] = {e.path.split('/', 1)[0] for e in manifest.files if '/' in e.path}
    found: list[str] = []
    for top in sorted(tops):
        base: Path = folder / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != '.cache']
            for name in filenames:
                if name.endswith(PART) or (name.startswith('put_') and name.endswith('_here')):
                    continue
                relative: str = (Path(dirpath) / name).relative_to(folder).as_posix()
                if relative not in known:
                    found.append(relative)
    return found


def check(manifest: Manifest, folder: Path, quick: bool = False) -> int:
    bad: int = 0
    for entry in manifest.files:
        state: str = status(entry, folder, quick)
        bad += state != 'ok'
        print(f'{state:11} {entry.path}')
    extras: list[str] = _extras(manifest, folder)
    if extras:
        print(f'{len(extras)} extra file(s) in {folder} the manifest does not name:')
        for path in extras:
            print(f'  extra       {path}')
    print(f'{len(manifest.files) - bad} of {len(manifest.files)} ok in {folder}' + (' (sizes only)' if quick else ''))
    return 1 if bad else 0


def install(manifest: Manifest, folder: Path, mirror: Path | None = None, quick: bool = False,
            token_getter: Callable[[], str | None] = get_token) -> int:
    """Every file that is not ok, in order: the mirror, the HF repo, the public source."""
    token: str | None = None
    token_asked: bool = False
    failed: list[str] = []
    fetched: int = 0
    for entry in manifest.files:
        target: Path = folder / entry.path
        if status(entry, folder, quick) == 'ok':
            continue
        if not entry.sha256:
            logger.error('%s: the manifest has no sha256 for it, run `sources --write` or `hash --write` first', entry.path)
            failed.append(entry.path)
            continue
        done: bool = False
        if mirror is not None and (mirror / entry.path).is_file():
            done = copy_verified(entry, mirror / entry.path, target)
        if not done and manifest.repo_ready:
            if not token_asked:
                token, token_asked = token_getter(), True
                if token is None:
                    logger.info('no Hugging Face token: the repo %s is skipped, only the public sources are tried', manifest.repo)
            if token is not None:
                done = download(entry, manifest.hf_url(entry), target, token)
        if not done and entry.source:
            done = download(entry, entry.source, target)
        if done:
            fetched += 1
            logger.info('%s: ok', entry.path)
        else:
            failed.append(entry.path)
            logger.error('%s: not installed', entry.path)
    if failed:
        print(f'{fetched} file(s) installed, {len(failed)} still missing in {folder}:')
        for path in failed:
            print(f'  {path}')
        return 1
    print(f'{fetched} file(s) installed, all {len(manifest.files)} ok in {folder}')
    return 0


def export(manifest: Manifest, folder: Path, dest: Path) -> int:
    """The verified set copied into dest with the manifest beside it (a mirror for an offline install)."""
    if check(manifest, folder) != 0:
        logger.error('not every file is ok, nothing exported')
        return 1
    copied: int = 0
    for entry in manifest.files:
        target: Path = dest / entry.path
        if status(entry, dest) == 'ok':
            continue
        if not copy_verified(entry, folder / entry.path, target):
            return 1
        copied += 1
    manifest.save(dest / 'models.json')
    print(f'{copied} file(s) copied, all {len(manifest.files)} in {dest}')
    return 0


def hash_files(manifest: Manifest, folder: Path, write: bool = False, add: list[str] | None = None) -> int:
    """Size and sha256 of the manifest's files present under folder; --write puts them in the
    manifest, --add adds entries (source null) for paths not in it yet."""
    for path in add or []:
        posix: str = path.replace('\\', '/')
        if manifest.get(posix) is None:
            manifest.files.append(Entry(posix))
    changed: int = 0
    for entry in manifest.files:
        target: Path = folder / entry.path
        if not target.is_file():
            continue
        size, sha256 = target.stat().st_size, file_sha256(target)
        state: str = 'new' if not entry.sha256 else 'unchanged' if (size, sha256) == (entry.size, entry.sha256) else 'CHANGED'
        print(f'{entry.path} {size} {sha256} {state}')
        if state != 'unchanged':
            changed += 1
            if write:
                entry.size, entry.sha256 = size, sha256
    if write:
        manifest.save()
        print(f'{changed} entr{"y" if changed == 1 else "ies"} written to {manifest.path}')
    elif changed:
        print(f'{changed} entr{"y" if changed == 1 else "ies"} differ(s); --write puts the values in the manifest')
    return 0


def sources(manifest: Manifest, write: bool = False, repo: bool = False, drop_mismatch: bool = False,
            token_getter: Callable[[], str | None] = get_token) -> int:
    """Every public source against its entry, by the hash and size the server reports (no download).
    An entry without a hash adopts the source's. --repo does the same for the HF repo's copies."""
    bad: int = 0
    for entry in manifest.files:
        if not entry.source:
            continue
        found: Resolved | None = resolve(entry.source)
        if found is None:
            print(f'unreachable  {entry.path}  {entry.source}')
            bad += 1
            continue
        if found.sha256 is None:
            state: str = 'no hash from source' if found.size is None else 'size ok' if found.size == entry.size else 'size MISMATCH'
        elif not entry.sha256:
            state = 'adopted' if write else 'adoptable'
            if write:
                entry.sha256 = found.sha256
                if found.size is not None:
                    entry.size = found.size
        elif found.sha256 == entry.sha256 and (found.size is None or found.size == entry.size):
            state = 'ok'
        else:
            state = 'MISMATCH'
            if write and drop_mismatch:
                entry.source = None
                state = 'MISMATCH, source dropped'
        bad += 'MISMATCH' in state        # 'no hash from source' (a small non-LFS file) is information, not a failure
        print(f'{state:20} {entry.path}')
    if repo:
        bad += _check_repo(manifest, token_getter)
    if write:
        manifest.save()
    return 1 if bad else 0


def _check_repo(manifest: Manifest, token_getter: Callable[[], str | None]) -> int:
    if not manifest.repo_ready:
        print(f'the manifest has no repo yet ({manifest.repo!r})')
        return 1
    token: str | None = token_getter()
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'} if token else {}
    url: str = f'https://huggingface.co/api/models/{manifest.repo}/tree/main?recursive=true'
    try:
        reply: Response = http(url, 'GET', headers)
    except OSError as e:
        print(f'repo {manifest.repo}: {e}')
        return 1
    if reply.status != 200 or reply.body is None:
        print(f'repo {manifest.repo}: HTTP {reply.status}')
        return 1
    tree: dict[str, dict] = {item['path']: item for item in json.load(reply.body) if item.get('type') == 'file'}
    bad: int = 0
    for entry in manifest.files:
        item: dict | None = tree.get(entry.path)
        if item is None:
            state: str = 'not in repo'
        else:
            lfs: dict = item.get('lfs') or {}
            oid: str = lfs.get('oid', '')
            size: int = int(lfs.get('size') or item.get('size') or 0)
            state = 'ok' if (not oid or oid == entry.sha256) and size == entry.size else 'MISMATCH'
            if not oid and size == entry.size:
                state = 'size ok'
        bad += state not in ('ok', 'size ok')
        print(f'{state:20} {manifest.repo}/{entry.path}')
    return bad


def upload(manifest: Manifest, folder: Path, hf_exe: Path | None = None,
           token_getter: Callable[[], str | None] = get_token, run: Callable[..., Any] = subprocess.run) -> int:
    """The verified set into the HF repo with hf.exe (huggingface_hub's CLI, in the ComfyUI venv)."""
    if not manifest.repo_ready:
        logger.error('the manifest has no repo yet (%r): create the private repo and put its id in the manifest', manifest.repo)
        return 1
    exe: Path | None = hf_exe or (HF_EXE if HF_EXE.is_file() else None) or (Path(shutil.which('hf')) if shutil.which('hf') else None)
    if exe is None or not Path(exe).is_file():
        logger.error('hf.exe not found (looked in %s and on the PATH): install ComfyUI first, or give --hf', HF_EXE)
        return 1
    if check(manifest, folder) != 0:
        logger.error('not every file is ok, nothing uploaded')
        return 1
    token: str | None = token_getter()
    if token is None:
        logger.error('no Hugging Face token (HF_TOKEN, %s, or the prompt)', HF_TOKEN_FILE)
        return 1
    command: list[str] = [str(exe), 'upload', manifest.repo, str(folder), '.', '--repo-type', 'model', '--private',
                          '--commit-message', 'SMILE models (modules.comfy.models upload)', '--token', token]
    for entry in manifest.files:
        command += ['--include', entry.path]        # one flag per file: hf takes a single pattern per --include
    print(' '.join('<token>' if arg == token else arg for arg in command))
    result = run(command, check=False)
    return int(result.returncode)


# ENTRY POINT

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='models', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)      # the options every command takes, after its name
    common.add_argument('--manifest', type=Path, default=MANIFEST, help='the manifest (default comfy/models.json)')
    common.add_argument('--models', type=Path, default=MODELS, help='the models folder (default <repo>\..\smile-models)')
    common.add_argument('-v', '--verbose', action='store_true', help='debug logging')
    sub = parser.add_subparsers(dest='command', required=True, parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))
    p = sub.add_parser('check', help='every manifest file ok / missing / wrong size / corrupt, extras listed')
    p.add_argument('--quick', action='store_true', help='sizes only, no hashing')
    p = sub.add_parser('install', help='fetch every file that is not ok: mirror, HF repo, public source')
    p.add_argument('--mirror', type=Path, help='a folder in the manifest layout to copy from first')
    p.add_argument('--quick', action='store_true', help='trust files whose size is right')
    p = sub.add_parser('export', help='verify, then copy the set with the manifest into DIR')
    p.add_argument('dir', type=Path)
    p = sub.add_parser('hash', help='size and sha256 of the files present; --write into the manifest')
    p.add_argument('--write', action='store_true')
    p.add_argument('--add', nargs='*', default=[], metavar='PATH', help='new entries (source null) for these paths')
    p = sub.add_parser('sources', help='the public sources (hash and size the server reports) against the manifest')
    p.add_argument('--write', action='store_true', help='entries without a hash adopt the source\'s')
    p.add_argument('--drop-mismatch', action='store_true', help='with --write: a source that holds another file becomes null')
    p.add_argument('--repo', action='store_true', help='also the HF repo\'s copies (needs the token)')
    p = sub.add_parser('upload', help='the verified set into the HF repo (hf.exe)')
    p.add_argument('--hf', type=Path, help='hf.exe to use (default ComfyUI\\.venv\\Scripts\\hf.exe)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s %(message)s', stream=sys.stderr)
    if not args.manifest.is_file():
        logger.error('no manifest at %s', args.manifest)
        return 1
    manifest: Manifest = Manifest.load(args.manifest)
    folder: Path = args.models
    if args.command == 'check':
        return check(manifest, folder, args.quick)
    if args.command == 'install':
        return install(manifest, folder, args.mirror, args.quick)
    if args.command == 'export':
        return export(manifest, folder, args.dir)
    if args.command == 'hash':
        return hash_files(manifest, folder, args.write, args.add)
    if args.command == 'sources':
        return sources(manifest, args.write, args.repo, args.drop_mismatch)
    if args.command == 'upload':
        return upload(manifest, folder, args.hf)
    return 2


if __name__ == '__main__':
    sys.exit(main())
