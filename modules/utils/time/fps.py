import time
import threading
from collections import deque

import numpy as np

WINDOW: float = 2.0     # s a rate is taken over: 120 frames at 60 fps


class FPS:
    """A rate of events (frames, captures) over the last `window` seconds, in events per second: its average, its
    lowest and its highest, from the gaps between their arrivals (`perf_counter`). The gap reaching into the window
    from before it counts, so a pause shows in the lowest rate for the window after it."""

    def __init__(self, window: float = WINDOW) -> None:
        self.window: float = window
        self.count_processed = -1
        self.count_dropped = 0
        self._stamps: deque[float] = deque()
        self.mutex = threading.Lock()

    def _gaps(self) -> np.ndarray:
        return np.diff(np.fromiter(self._stamps, dtype=float, count=len(self._stamps)))

    def get_rate_minimum(self) -> float:
        with self.mutex:
            if len(self._stamps) < 2:
                return 0.0
            longest: float = float(self._gaps().max())
            return 1.0 / longest if longest > 0.0 else 1000000.0

    def get_rate_maximum(self) -> float:
        with self.mutex:
            if len(self._stamps) < 2:
                return 0.0
            shortest: float = float(self._gaps().min())
            return 1.0 / shortest if shortest > 0.0 else 1000000.0

    def get_rate_average(self) -> float:
        """The gaps in the window over the time they span: one over their mean."""
        with self.mutex:
            if len(self._stamps) < 2:
                return 0.0
            span: float = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0.0 else 1000000.0

    def get_count_processed(self) -> int:
        return self.count_processed

    def get_count_dropped(self) -> int:
        return self.count_dropped

    def processed(self) -> None:
        with self.mutex:
            # perf_counter, not time.time: Windows' ~16 ms time() resolution would drown the
            # frame-gap jitter these gaps measure
            now: float = time.perf_counter()
            self.count_processed += 1
            self._stamps.append(now)
            start: float = now - self.window
            while len(self._stamps) > 2 and self._stamps[1] <= start:
                self._stamps.popleft()

    def dropped(self) -> None:
        self.count_dropped += 1

    def reset(self) -> None:
        with self.mutex:
            self._stamps.clear()
            self.count_dropped = 0
            self.count_processed = -1
