"""Compare two timing reports of `launcher.py -timing`: a baseline and a new build.

    python scripts/timing_compare.py logs/timing_white_space_<base>.json logs/timing_white_space_<new>.json

A timer passes when its mean and its p95 are at most `--tolerance` percent above the baseline's.
The exit code is 1 when a timer fails or is missing from the new report.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _change(base: float, new: float) -> float:
    return (new - base) / base * 100.0 if base > 0 else 0.0


def main() -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('baseline', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--tolerance', type=float, default=5.0, help='percent a timer may be slower (default 5)')
    args = parser.parse_args()

    base, new = _load(args.baseline), _load(args.new)
    if base['app'] != new['app']:
        print(f"Different apps: {base['app']} and {new['app']}")
        return 1

    print(f"{base['app']}: {base['commit']} -> {new['commit']}")
    for name in sorted(set(base['packages']) | set(new['packages'])):
        old_v, new_v = base['packages'].get(name, '-'), new['packages'].get(name, '-')
        if old_v != new_v:
            print(f"  {name:<22} {old_v} -> {new_v}")
    print()

    header = f"{'timer':<22} {'count':^13} {'mean ms':^17} {'change':>7} {'p95 ms':^17} {'change':>7}  result"
    print(header)
    print('-' * len(header))
    failed = False
    for name in sorted(set(base['timers']) | set(new['timers'])):
        b, n = base['timers'].get(name), new['timers'].get(name)
        if b is None or n is None:
            print(f"{name:<22} {'only in ' + ('new' if b is None else 'baseline'):>13}")
            failed = failed or n is None
            continue
        mean_change, p95_change = _change(b['mean'], n['mean']), _change(b['p95'], n['p95'])
        ok = mean_change <= args.tolerance and p95_change <= args.tolerance
        failed = failed or not ok
        print(f"{name:<22} {b['count']:>6}/{n['count']:<6} {b['mean']:>8.3f}/{n['mean']:<8.3f} {mean_change:>+6.1f}% "
              f"{b['p95']:>8.3f}/{n['p95']:<8.3f} {p95_change:>+6.1f}%  {'ok' if ok else 'SLOWER'}")

    print()
    print('FAIL' if failed else 'PASS', f"(tolerance {args.tolerance:.0f}%)")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
