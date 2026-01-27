import threading
import numpy as np
from typing import Optional


class PerformanceTimer:
    """Track and report performance timing statistics over a rolling window."""

    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m',
    }

    def __init__(
        self,
        name: str = "Performance",
        sample_count: int = 100,
        report_interval: Optional[int] = None,
        color: Optional[str] = None
    ) -> None:
        """Initialize performance timer.

        Args:
            name: Name to display in reports
            sample_count: Number of samples to track for statistics
            report_interval: Print report every N samples (None = at sample_count)
            color: Optional color name ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')
        """
        self.name = name
        self.sample_count = sample_count
        self.report_interval = report_interval or sample_count
        self.times: np.ndarray = np.zeros(sample_count, dtype=float)
        self.current_index: int = 0
        self.count: int = 0
        self.mutex = threading.Lock()

        # Set color codes
        if color and color.lower() in self.COLORS:
            self.color_start = self.COLORS[color.lower()]
            self.color_end = self.COLORS['reset']
        else:
            self.color_start = ''
            self.color_end = ''

    def add_time(self, time_ms: float, report: bool = True) -> None:
        """Add timing sample and optionally print report.

        Args:
            time_ms: Timing measurement in milliseconds
            report: Whether to print periodic reports
        """
        with self.mutex:
            self.times[self.current_index] = time_ms
            self.current_index = (self.current_index + 1) % self.sample_count
            self.count += 1

            should_report = self.count % self.report_interval == 0

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
        print(f"{self.color_start}{self.name}: avg={avg:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms{self.color_end}")

    def reset(self) -> None:
        """Reset all timing data."""
        with self.mutex:
            self.times.fill(0)
            self.current_index = 0
            self.count = 0