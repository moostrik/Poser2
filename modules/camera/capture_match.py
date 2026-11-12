"""What the syncs that bring the parts of one capture together share: a part's values of one camera kept by
capture time and found within `CAPTURE_TOLERANCE` (`CaptureStream`), and the count of parts a capture was made
without (`MissedParts`). The rules of when a capture goes out are each sync's own: the live one puts latency
first (`modules.inference.CaptureSync`), the recorder's completeness (`RecordingSync`)."""

import logging
import time
from collections import deque
from threading import Lock
from typing import Iterator

from .frame import CAPTURE_TOLERANCE

logger = logging.getLogger(__name__)

WARN_INTERVAL: float = 1.0      # s: the shortest time between two warnings of one sync


class CaptureStream[T]:
    """One part of one camera: the newest `kept` values with their capture times, oldest first. Not locked: the
    sync that owns it holds its own lock around it."""

    def __init__(self, kept: int) -> None:
        self._values: deque[tuple[float, T]] = deque(maxlen=kept)

    def __iter__(self) -> Iterator[tuple[float, T]]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, timestamp: float, value: T) -> bool:
        """Keep `value`; True when the oldest went to make room, a value no capture took."""
        full: bool = len(self._values) == self._values.maxlen
        self._values.append((timestamp, value))
        return full

    def at(self, timestamp: float) -> T | None:
        """The value captured at `timestamp`; None when it has not come in."""
        for value_time, value in self._values:
            if abs(value_time - timestamp) <= CAPTURE_TOLERANCE:
                return value
        return None

    def before(self, timestamp: float) -> T | None:
        """The newest value captured before `timestamp`; None when there is none."""
        for value_time, value in reversed(self._values):
            if value_time < timestamp:
                return value
        return None

    def newest(self) -> float | None:
        """The capture time of the newest value; None while there is none."""
        return self._values[-1][0] if self._values else None

    def keeps_up(self, last: float) -> bool:
        """The newest is as new as `last`, the camera's last capture: the next one is on its way. False while
        there is none, and while the stream runs a frame or more behind."""
        newest: float | None = self.newest()
        return newest is not None and newest >= last - CAPTURE_TOLERANCE

    def take_through(self, timestamp: float) -> int:
        """Remove what was captured up to `timestamp`; the count of those captured before it, which no capture
        will take any more."""
        missed: int = 0
        while self._values and self._values[0][0] <= timestamp + CAPTURE_TOLERANCE:
            value_time, _ = self._values.popleft()
            if value_time < timestamp - CAPTURE_TOLERANCE:
                missed += 1
        return missed

    def clear(self) -> None:
        self._values.clear()


class MissedParts:
    """The parts a sync made a capture without, or dropped as no capture's, per camera and part. `count` only
    counts; `warn`, on the owner's thread, logs the rise as a warning naming the sync, the camera and the part, at
    most once in `WARN_INTERVAL`. What is counted before the sync's first capture is counted apart and not warned
    of (`at_start`)."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._lock: Lock = Lock()
        self._counts: dict[tuple[int, str], int] = {}
        self._at_start: dict[str, int] = {}
        self._started: bool = False
        self._warned: float = float('-inf')
        self._unwarned: dict[tuple[int, str], int] = {}

    def started(self) -> None:
        """The sync's first capture went out: what is counted from here is warned of."""
        with self._lock:
            self._started = True

    def reset(self) -> None:
        """A new start (a new recording): the counts from zero, the start counted apart again."""
        with self._lock:
            self._counts.clear()
            self._unwarned.clear()
            self._at_start.clear()
            self._started = False

    def count(self, cam_id: int, part: str, missed: int = 1) -> None:
        if missed <= 0:
            return
        with self._lock:
            if not self._started:
                self._at_start[part] = self._at_start.get(part, 0) + missed
                return
            key: tuple[int, str] = (cam_id, part)
            self._counts[key] = self._counts.get(key, 0) + missed
            self._unwarned[key] = self._unwarned.get(key, 0) + missed

    def warn(self) -> None:
        """Log what was counted since the last warning, at most once in `WARN_INTERVAL`: called by the owner's own
        thread, never where the count is made, which can be a camera's thread under a sync's lock."""
        with self._lock:
            now: float = time.perf_counter()
            if not self._unwarned or now - self._warned < WARN_INTERVAL:
                return
            self._warned = now
            unwarned: dict[tuple[int, str], int] = self._unwarned
            self._unwarned = {}
        logger.warning('%s: %s', self.name, ', '.join(f'cam {cam} {part} missed {n}' for (cam, part), n in sorted(unwarned.items())))

    def total(self, part: str) -> int:
        """Missed of `part` over every camera since the first capture."""
        with self._lock:
            return sum(n for (_, name), n in self._counts.items() if name == part)

    def at_start(self, part: str) -> int:
        """Missed of `part` before the first capture: a few at every start."""
        with self._lock:
            return self._at_start.get(part, 0)
