"""Timing sessions: the samples of every PerformanceTimer over a fixed stretch of a run, as JSON.

`launcher.py -timing <seconds>` runs one: it waits out the warm-up, keeps every sample for the
given seconds, writes the report next to the log and stops the app. `scripts/timing_compare.py`
compares two reports.
"""

import json
import logging
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from threading import Event

from .PerformanceTimer import PerformanceTimer

logger = logging.getLogger(__name__)

# The packages whose versions a report names: the ones an upgrade can make faster or slower
TIMED_PACKAGES: tuple[str, ...] = (
    'numpy', 'opencv-python', 'torch', 'tensorrt',
    'onnxruntime-gpu', 'depthai', 'PyOpenGL', 'PyOpenGL-accelerate', 'glfw', 'OneEuroFilter',
)


def package_versions() -> dict[str, str]:
    """Installed versions of `TIMED_PACKAGES`; `tensorrt` also matches `tensorrt-cu12` and `-cu13`."""
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name: str = dist.metadata['Name'] or ''
        for wanted in TIMED_PACKAGES:
            if name.lower() == wanted.lower() or (wanted == 'tensorrt' and name.lower().startswith('tensorrt-cu')
                                                  and not name.lower().endswith(('-libs', '-bindings'))):
                versions[name] = dist.version
    return dict(sorted(versions.items()))


def git_commit() -> str:
    """The checked-out commit, with `-dirty` when the tree has changes; empty without git."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=root,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''
    return commit


def run_session(warmup: float, duration: float, stop: Event) -> dict[str, dict[str, float]] | None:
    """Wait `warmup` seconds, then keep every timer's samples for `duration` seconds.

    Returns the statistics per timer, or None when `stop` is set before the session ends.
    """
    if stop.wait(warmup):
        return None
    logger.info("Timing session started: %.0f s", duration)
    PerformanceTimer.start_session()
    stopped = stop.wait(duration)
    stats = PerformanceTimer.stop_session()
    if stopped:
        logger.warning("Timing session stopped early, no report written")
        return None
    return stats


def write_report(log_dir: Path, app: str, simulation: bool, warmup: float, duration: float,
                 stats: dict[str, dict[str, float]]) -> Path:
    """Write the session's report as `timing_<app>_<date>_<time>.json` in `log_dir`."""
    path = log_dir / f"timing_{app}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        'app': app,
        'simulation': simulation,
        'warmup_s': warmup,
        'duration_s': duration,
        'commit': git_commit(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': package_versions(),
        'timers': stats,
    }
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    logger.info("Timing report written: %s", path)
    return path
