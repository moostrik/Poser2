"""deep_flow's timing runs, back to back, each on the startup files with settings changed by path.

Each run copies the startup setup and flow files to `trial.json` in their folders with its changes, points
`.startup.json` at them, runs `launcher.py -app deep_flow -sim -ni -timing 60 -warmup 10` and puts the pointer
and the folders back as they were, so the operator's files are never written. The reports land in `logs/` as
every timing run's do; the script prints their names and the fluid's timers side by side.

    python scripts/trial_runs.py --run base --run "visc:fluid.viscosity.viscosity=0.0001" [--seconds 60]
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / 'apps' / 'deep_flow' / 'data' / 'settings'
POINTER = SETTINGS / '.startup.json'
GROUPS = ('setup', 'flow')
DEFAULT_STARTUP = 'studio'
TRIAL = 'trial'
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'


def changed(tree: dict, changes: list[str]) -> dict:
    """The tree with each `path=value` set; a path's first part may name a top-level group or the group holding it."""
    for change in changes:
        path, value = change.split('=', 1)
        parts = path.split('.')
        node = _find(tree, parts[0])
        for part in parts[1:-1]:
            node = node[part]
        current = node[parts[-1]]
        if isinstance(current, (list, dict)):
            node[parts[-1]] = json.loads(value)         # a list or a group as JSON: [] or ["M1"]
        elif isinstance(current, bool):
            node[parts[-1]] = value in ('1', 'true', 'True')
        else:
            node[parts[-1]] = type(current)(value)
    return tree


def _find(node: dict, key: str) -> dict:
    """The first dict under `key`, breadth first."""
    queue = [node]
    while queue:
        item = queue.pop(0)
        if key in item and isinstance(item[key], dict):
            return item[key]
        queue += [v for v in item.values() if isinstance(v, dict)]
    raise KeyError(key)


def report_names() -> set[str]:
    return {p.name for p in (ROOT / 'logs').glob('timing_deep_flow_*.json')}


def run(label: str, changes: list[str], seconds: int, warmup: int) -> Path | None:
    pointer: str | None = POINTER.read_text(encoding='utf-8') if POINTER.exists() else None
    startup: dict[str, str] = json.loads(pointer) if pointer else {}
    # the setup's file holds the root's fields beside `setup`; the flow joins it as `flow`, so a path finds either
    tree: dict = json.loads((SETTINGS / 'setup' / f"{startup.get('setup', DEFAULT_STARTUP)}.json").read_text(encoding='utf-8'))
    tree['flow'] = json.loads((SETTINGS / 'flow' / f"{startup.get('flow', DEFAULT_STARTUP)}.json").read_text(encoding='utf-8'))
    tree = changed(tree, changes)
    trials = {group: SETTINGS / group / f'{TRIAL}.json' for group in GROUPS}
    before = report_names()
    try:
        trials['flow'].write_text(json.dumps(tree.pop('flow'), indent=2), encoding='utf-8')
        trials['setup'].write_text(json.dumps(tree, indent=2), encoding='utf-8')
        POINTER.write_text(json.dumps({group: TRIAL for group in GROUPS}, indent=2), encoding='utf-8')
        print(f'{label}: {", ".join(changes) or "the startup files"}', flush=True)
        subprocess.run([str(PYTHON), 'launcher.py', '-app', 'deep_flow', '-sim', '-ni',
                        '-timing', str(seconds), '-warmup', str(warmup)], cwd=ROOT, check=False)
    finally:
        if pointer is None:
            POINTER.unlink(missing_ok=True)
        else:
            POINTER.write_text(pointer, encoding='utf-8')
        for trial in trials.values():
            trial.unlink(missing_ok=True)
    new = sorted(report_names() - before)
    return ROOT / 'logs' / new[-1] if new else None


def main() -> None:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', action='append', required=True, help='label[:path=value,...]')
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    reports: list[tuple[str, Path | None]] = []
    for item in args.run:
        label, _, rest = item.partition(':')
        reports.append((label, run(label, [c for c in rest.split(',') if c], args.seconds, args.warmup)))
    timers: dict[str, dict[str, tuple[float, float]]] = {}
    for label, report in reports:
        print(f'{label}: {report.name if report else "no report"}')
        if report is None:
            continue
        data = json.loads(report.read_text(encoding='utf-8'))
        for name, stats in data.get('timers', {}).items():
            if name.startswith('GPU fluid') or name == 'Render interval':
                timers.setdefault(name, {})[label] = (stats.get('mean', 0.0), stats.get('p95', 0.0))
    labels = [label for label, _ in reports]
    print(f"{'timer':<34}" + ''.join(f'{label:>16}' for label in labels))
    for name in sorted(timers):
        print(f'{name:<34}' + ''.join(f"{'%.2f / %.2f' % timers[name][l] if l in timers[name] else '-':>16}" for l in labels))


if __name__ == '__main__':
    sys.exit(main())
