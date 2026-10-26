"""Records what a ComfyUI install consists of, without changing it: the core commit, every node
pack (version, git commit when it has one, local changes), every model file (size, sha256) and
the venv's packages. The clean installer's locks are made from this output.

Run from the repo root with any Python 3, on the PC whose install is the reference:

    py -3.12 comfy\\inventory.py                  (ComfyUI\\ next to comfy\\)
    py -3.12 comfy\\inventory.py -c D:\\ComfyUI    (another install)

Packs without git are compared with their Comfy Registry archive (needs internet, --offline
skips it), so hand edits show as diffs. Writes output\\inventory\\inventory.json and
pip_freeze.txt. Hashing ~25 GB takes a few minutes.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0  # CREATE_NO_WINDOW


def run(args: list[str], cwd: Path | None = None) -> str:
    """A command's output, or '' when it fails."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True,
                              creationflags=NO_WINDOW).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def git_state(path: Path) -> dict:
    """Commit, remote and local changes of a git checkout; empty when it is none."""
    if not (path / '.git').exists():
        return {}
    return {'commit': run(['git', 'rev-parse', 'HEAD'], path),
            'remote': run(['git', 'remote', 'get-url', 'origin'], path),
            'branch': run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], path),
            'changes': run(['git', 'status', '--porcelain'], path).splitlines(),
            'diff': run(['git', 'diff'], path)}


def pack_version(path: Path) -> str:
    """The version in a pack's pyproject.toml (what the Comfy Registry publishes)."""
    toml = path / 'pyproject.toml'
    if toml.exists():
        found = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding='utf-8', errors='replace'), re.M)
        if found:
            return found.group(1)
    return ''


def registry_diff(path: Path, version: str) -> dict:
    """How a pack differs from its Comfy Registry archive of the same version: the hand edits
    a pack without git carries. Text diffs for changed .py files, names for the rest."""
    try:
        with urllib.request.urlopen(f'https://api.comfy.org/nodes/{path.name}/versions/{version}', timeout=30) as r:
            url: str = json.load(r)['downloadUrl']
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / ('pack.tar.gz' if url.endswith('.tar.gz') else 'pack.zip')
            urllib.request.urlretrieve(url, archive)
            original = Path(temp) / 'pack'
            if archive.suffix == '.zip':
                zipfile.ZipFile(archive).extractall(original)
            else:
                tarfile.open(archive).extractall(original, filter='data')
            return compare(original, path)
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def compare(original: Path, installed: Path) -> dict:
    def files(root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}
    a, b = files(original), files(installed)
    changed: dict[str, str] = {}
    for name in sorted(a & b):
        old, new = (original / name).read_bytes(), (installed / name).read_bytes()
        if old != new:
            changed[name] = ''.join(difflib.unified_diff(
                old.decode('utf-8', 'replace').splitlines(True), new.decode('utf-8', 'replace').splitlines(True),
                f'a/{name}', f'b/{name}')) if name.endswith('.py') else 'binary or data file differs'
    return {'changed': changed, 'added': sorted(b - a), 'removed': sorted(a - b)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def models(folder: Path, hashed: bool) -> list[dict]:
    """Every file under a models folder, symlinks and junctions followed."""
    found: list[dict] = []
    for dirpath, _, files in os.walk(folder, followlinks=True):
        for name in sorted(files):
            path = Path(dirpath) / name
            print(f'  {path.relative_to(folder)}', flush=True)
            found.append({'path': path.relative_to(folder).as_posix(), 'size': path.stat().st_size,
                          'sha256': sha256(path) if hashed else ''})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--comfy', type=Path, default=ROOT / 'ComfyUI', help='the ComfyUI install')
    parser.add_argument('-o', '--out', type=Path, default=ROOT / 'output' / 'inventory', help='output folder')
    parser.add_argument('--offline', action='store_true', help='no comparison with the Comfy Registry archives')
    parser.add_argument('--no-hash', action='store_true', help='sizes only, no sha256 (fast)')
    args = parser.parse_args()
    comfy: Path = args.comfy.resolve()
    if not (comfy / 'main.py').exists():
        sys.exit(f'no ComfyUI at {comfy}')
    args.out.mkdir(parents=True, exist_ok=True)

    print(f'ComfyUI at {comfy}')
    packs: dict[str, dict] = {}
    for path in sorted(p for p in (comfy / 'custom_nodes').iterdir() if p.is_dir() and p.name != '__pycache__'):
        pack = {'version': pack_version(path), 'link': path.is_symlink() or path.is_junction(), **git_state(path)}
        if not pack.get('commit') and pack['version'] and not args.offline:
            print(f'comparing {path.name} {pack["version"]} with the Comfy Registry', flush=True)
            pack['registry_diff'] = registry_diff(path, pack['version'])
        packs[path.name] = pack

    model_dirs = [comfy / 'models', Path.home() / '.cache' / 'huggingface' / 'hub']
    print('models' + (' (sizes only)' if args.no_hash else ', hashing'))
    found_models: dict[str, list[dict]] = {}
    for folder in model_dirs:
        if folder.exists():
            found_models[str(folder)] = models(folder, not args.no_hash)

    python = comfy / '.venv' / 'Scripts' / 'python.exe'
    if not python.exists():
        python = comfy / '.venv' / 'bin' / 'python'
    freeze = run([str(python), '-m', 'pip', 'freeze', '--all'])
    (args.out / 'pip_freeze.txt').write_text(freeze + '\n', encoding='utf-8')

    inventory = {
        'made': datetime.now().isoformat(timespec='seconds'),
        'host': os.environ.get('COMPUTERNAME', ''),
        'gpu': run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
        'python': run([str(python), '--version']),
        'torch': run([str(python), '-c', 'import torch; print(torch.__version__, torch.version.cuda)']),
        'comfyui': git_state(comfy),
        'packs': packs,
        'models': found_models,
        'smile': git_state(ROOT),
    }
    (args.out / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    total = sum(m['size'] for files in found_models.values() for m in files)
    print(f'{len(packs)} packs, {sum(map(len, found_models.values()))} model files ({total / 1e9:.1f} GB) -> {args.out}')


if __name__ == '__main__':
    main()
