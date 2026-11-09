"""deep_flow's timing runs, back to back, each on the preset with settings changed by path.

Each run copies the startup preset to `trial.json` beside it with its changes, points `.ui_preset.json` at it,
runs `launcher.py -app deep_flow -sim -ni -timing 60 -warmup 10` and puts the pointer and the folder back as
they were, so the operator's presets are never written. The reports land in `logs/` as every timing run's do;
the script prints their names and the fluid's timers side by side.

    python scripts/trial_runs.py --run base --run "visc:fluid.velocity.viscosity=0.0001" [--seconds 60]
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / 'apps' / 'deep_flow' / 'data' / 'settings'
POINTER = SETTINGS / '.ui_preset.json'
TRIAL = 'trial'
PYTHON = ROOT / '.venv-oak4' / 'Scripts' / 'python.exe'


def changed(preset: dict, changes: list[str]) -> dict:
    """The preset with each `path=value` set; a path's first part may name a top-level group or the group holding it."""
    for change in changes:
        path, value = change.split('=', 1)
        parts = path.split('.')
        node = _find(preset, parts[0])
        for part in parts[1:-1]:
            node = node[part]
        current = node[parts[-1]]
        if isinstance(current, (list, dict)):
            node[parts[-1]] = json.loads(value)         # a list or a group as JSON: [] or ["M1"]
        elif isinstance(current, bool):
            node[parts[-1]] = value in ('1', 'true', 'True')
        else:
            node[parts[-1]] = type(current)(value)
    return preset


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
    startup = json.loads(POINTER.read_text(encoding='utf-8'))['startup_preset']
    preset = json.loads((SETTINGS / f'{startup}.json').read_text(encoding='utf-8'))
    trial = SETTINGS / f'{TRIAL}.json'
    pointer = POINTER.read_text(encoding='utf-8')
    before = report_names()
    try:
        trial.write_text(json.dumps(changed(preset, changes), indent=2), encoding='utf-8')
        POINTER.write_text(json.dumps({'startup_preset': TRIAL}, indent=2), encoding='utf-8')
        print(f'{label}: {", ".join(changes) or "the preset"}', flush=True)
        subprocess.run([str(PYTHON), 'launcher.py', '-app', 'deep_flow', '-sim', '-ni',
                        '-timing', str(seconds), '-warmup', str(warmup)], cwd=ROOT, check=False)
    finally:
        POINTER.write_text(pointer, encoding='utf-8')
        trial.unlink(missing_ok=True)
    new = sorted(report_names() - before)
    return ROOT / 'logs' / new[-1] if new else None


def main() -> None:
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
