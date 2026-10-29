import threading
import weakref
import numpy as np
from typing import ClassVar

import logging
logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Track and report performance timing statistics over a rolling window.

    Every timer also takes part in the timing session: between `start_session()` and
    `stop_session()` each sample is kept, and `stop_session()` returns the statistics of the
    whole session per timer name (timers that share a name are pooled).
    """

    _registry: ClassVar[weakref.WeakSet['PerformanceTimer']] = weakref.WeakSet()
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()
    _session_active: ClassVar[bool] = False

    # ANSI color codes
    COLORS = {
        'red':      '\033[91m',
        'green':    '\033[92m',
        'yellow':   '\033[93m',
        'blue':     '\033[94m',
        'magenta':  '\033[95m',
        'cyan':     '\033[96m',
        'white':    '\033[97m',
        'reset':    '\033[0m',
    }

    def __init__(
        self,
        name: str = "Performance",
        sample_count: int = 100,
        omit_init: int = 0,
        report_interval: int | None = None,
        color: str | None = None,
    ) -> None:
        """Initialize performance timer.

        Args:
            name: Name to display in reports
            sample_count: Number of samples to track for statistics
            report_interval: Print report every N samples (None = at sample_count)
            color: Optional color name ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')
            omit_init: Number of initial samples to omit from statistics (default 0)
        """
        self.name = name
        self.sample_count = sample_count
        self.report_interval = report_interval or sample_count
        self.times: np.ndarray = np.zeros(sample_count, dtype=float)
        self.current_index: int = 0
        self.omit_init = omit_init

        self.mutex = threading.Lock()
        self.count: int = 0
        self.omit_count: int = 0
        self._session_times: list[float] = []

        with PerformanceTimer._registry_lock:
            PerformanceTimer._registry.add(self)

        # Set color codes
        if color and color.lower() in self.COLORS:
            self.color_start = self.COLORS[color.lower()]
            self.color_end = self.COLORS['reset']
        else:
            self.color_start = ''
            self.color_end = ''

    def add_time(self, time_ms: float, report: bool = True) -> None:
        """Add timing sample and optionally print report, omitting the first omit_init samples from recording."""
        with self.mutex:
            if PerformanceTimer._session_active:
                self._session_times.append(time_ms)
            if self.omit_count < self.omit_init:
                self.omit_count += 1
                return
            self.times[self.current_index] = time_ms
            self.current_index = (self.current_index + 1) % self.sample_count
            self.count += 1
            should_report = (self.count - self.omit_init) > 0 and (self.count - self.omit_init) % self.report_interval == 0

        # Print report OUTSIDE the lock to avoid deadlock
        if report and should_report:
            self._print_report()

    def get_average(self) -> float:
        """Get average time over collected samples."""
        with self.mutex:
            if self.count == 0:
                return 0.0
            if self.count < self.sample_count:
                return float(np.mean(self.times[:self.count]))
            return float(np.mean(self.times))

    def get_minimum(self) -> float:
        """Get minimum time over collected samples."""
        with self.mutex:
            if self.count == 0:
                return 0.0
            if self.count < self.sample_count:
                return float(np.min(self.times[:self.count]))
            return float(np.min(self.times))

    def get_maximum(self) -> float:
        """Get maximum time over collected samples."""
        with self.mutex:
            if self.count == 0:
                return 0.0
            if self.count < self.sample_count:
                return float(np.max(self.times[:self.count]))
            return float(np.max(self.times))

    def _print_report(self) -> None:
        """Print timing statistics (called internally when report_interval is reached)."""
        avg = self.get_average()
        min_time = self.get_minimum()
        max_time = self.get_maximum()
        logger.info(f"{self.color_start}{self.name}: avg={avg:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms{self.color_end}")

    def reset(self) -> None:
        """Reset all timing data."""
        with self.mutex:
            self.times.fill(0)
            self.current_index = 0
            self.count = 0

    @classmethod
    def start_session(cls) -> None:
        """Start keeping every sample of every timer, dropping those of an earlier session."""
        with cls._registry_lock:
            timers = list(cls._registry)
        for timer in timers:
            with timer.mutex:
                timer._session_times = []
        cls._session_active = True

    @classmethod
    def stop_session(cls) -> dict[str, dict[str, float]]:
        """End the session; return count, mean, p50, p95, p99 and max (ms) per timer name."""
        cls._session_active = False
        with cls._registry_lock:
            timers = list(cls._registry)
        pooled: dict[str, list[float]] = {}
        for timer in timers:
            with timer.mutex:
                samples, timer._session_times = timer._session_times, []
            if samples:
                pooled.setdefault(timer.name.strip(), []).extend(samples)
        stats: dict[str, dict[str, float]] = {}
        for name in sorted(pooled):
            times = np.asarray(pooled[name], dtype=float)
            p50, p95, p99 = np.percentile(times, [50, 95, 99])
            stats[name] = {
                'count': int(times.size),
                'mean': float(times.mean()),
                'p50': float(p50),
                'p95': float(p95),
                'p99': float(p99),
                'max': float(times.max()),
            }
        return stats
