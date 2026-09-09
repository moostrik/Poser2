"""Master clock — paces and measures the render loop.

`next_tick()` is the single authority for both timing and measurement: it blocks until the next
frame deadline (high-resolution monotonic `perf_counter`, with a short busy-spin tail for
sub-millisecond accuracy), then measures `dt` and returns the `Tick`.

Deliberately carries no musical time: the show's musical clock is the playhead bar (one
revolution at `low_rpm`) — states count bars, the sound side rides them. Debug patterns
animate on plain seconds (`tick.time`) with their own rate knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, sleep

from modules.settings import BaseSettings, Field

# Seconds before each deadline to stop coarse-sleeping and busy-spin instead.
# Trades a brief spin (~this long, per frame) for sub-100 µs deadline accuracy.
_SPIN_MARGIN: float = 0.001


class ClockSettings(BaseSettings):
    time: Field[float] = Field(0.0, access=Field.READ, description="Elapsed wall-clock time (s)")


@dataclass
class Tick:
    """Immutable clock snapshot produced once per render tick."""
    time: float   # monotonic elapsed seconds since the first tick
    dt:   float   # seconds elapsed since the previous tick


class Clock:
    """Paces the render loop at light_rate Hz and measures each tick.

    Owns the per-frame cadence: callers loop on the blocking ``next_tick()``.
    """

    def __init__(self, settings: ClockSettings, rate: float) -> None:
        self._settings  = settings
        self._interval: float = 1.0 / rate
        self._start: float | None = None   # baselines set lazily on the first tick
        self._last:  float = 0.0
        self._next:  float = 0.0

    @property
    def interval(self) -> float:
        return self._interval

    def next_tick(self) -> Tick:
        """Block until the next frame deadline, then measure and return the Tick.

        The first call establishes the timing baselines and returns immediately
        (dt = 0), so the construction→start gap produces no startup dt spike.
        """
        if self._start is None:
            now = perf_counter()
            self._start = now
            self._last  = now
            self._next  = now
        else:
            self._next += self._interval
            self._wait_until(self._next)
            now = perf_counter()
            if now - self._next > self._interval:
                # Severe overrun — resync rather than burst a run of catch-up frames.
                self._next = now

        dt = now - self._last
        self._last = now

        t = Tick(time=now - self._start, dt=dt)
        self._settings.time = t.time
        return t

    @staticmethod
    def _wait_until(deadline: float) -> None:
        """Sleep up to _SPIN_MARGIN before the deadline, then busy-spin the remainder."""
        coarse = deadline - perf_counter() - _SPIN_MARGIN
        if coarse > 0:
            sleep(coarse)
        while perf_counter() < deadline:
            pass
