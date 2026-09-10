"""Master clock — paces and measures the render loop.

`next_tick()` is the single authority for both timing and measurement: it blocks until the next
frame deadline (high-resolution monotonic `perf_counter`, with a short busy-spin tail for
sub-millisecond accuracy), then measures `dt` and returns the `Tick`.

Deliberately carries no musical time: the show's musical clock is the playhead bar (one
revolution at `beam_rpm`) — states count bars, the sound side rides them. Debug patterns
animate on plain seconds (`tick.time`) with their own rate knobs.

Accuracy (measured ~40 µs mean lateness idle) depends on the calling thread's OS priority
(the Conductor raises it) and on the GIL: a pure-Python thread holding it makes a sleeping
thread up to one switch interval (5 ms) late. Lowering the switch interval process-wide is
not an option — it starves the pose pipeline (see launcher.py). `late_max_ms` / `dt_max_ms`
/ `busy_max_ms` / `overruns` report the real-world result.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, sleep

from modules.settings import BaseSettings, Field

# Seconds before each deadline to stop coarse-sleeping and busy-spin instead.
# Trades a brief spin (~this long, per frame) for sub-100 µs deadline accuracy.
_SPIN_MARGIN: float = 0.001


# Diagnostics publish interval (s): running maxima are pushed to the settings and reset this often.
_STATS_INTERVAL: float = 1.0


class ClockSettings(BaseSettings):
    time:        Field[float] = Field(0.0, access=Field.READ, description="Elapsed wall-clock time (s)")
    late_max_ms: Field[float] = Field(0.0, access=Field.READ, description="Worst tick lateness vs. its deadline over the last second (ms)")
    dt_max_ms:   Field[float] = Field(0.0, access=Field.READ, description="Longest tick interval over the last second (ms)")
    busy_max_ms: Field[float] = Field(0.0, access=Field.READ, description="Longest per-tick work (tick return → next call) over the last second (ms) — the conductor's GIL share is busy/interval")
    overruns:    Field[int]   = Field(0,   access=Field.READ, description="Ticks that ran more than one interval late and resynced (cumulative)")


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
        # Diagnostics: running maxima since the last publish, and the cumulative resync count.
        self._late_max: float = 0.0
        self._dt_max:   float = 0.0
        self._busy_max: float = 0.0
        self._overruns: int   = 0
        self._stats_at: float = 0.0

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
            self._start    = now
            self._last     = now
            self._next     = now
            self._stats_at = now
        else:
            busy = perf_counter() - self._last      # the caller's work since the previous tick
            if busy > self._busy_max:
                self._busy_max = busy
            self._next += self._interval
            self._wait_until(self._next)
            now  = perf_counter()
            late = now - self._next
            if late > self._late_max:
                self._late_max = late
            if late > self._interval:
                # Severe overrun — resync rather than burst a run of catch-up frames.
                self._next = now
                self._overruns += 1

        dt = now - self._last
        self._last = now
        if dt > self._dt_max:
            self._dt_max = dt

        t = Tick(time=now - self._start, dt=dt)
        self._settings.time = t.time
        if now - self._stats_at >= _STATS_INTERVAL:
            self._publish_stats(now)
        return t

    def _publish_stats(self, now: float) -> None:
        """Push the running maxima to the settings (once per _STATS_INTERVAL) and reset them."""
        self._settings.late_max_ms = self._late_max * 1000.0
        self._settings.dt_max_ms   = self._dt_max * 1000.0
        self._settings.busy_max_ms = self._busy_max * 1000.0
        self._settings.overruns    = self._overruns
        self._late_max = 0.0
        self._dt_max   = 0.0
        self._busy_max = 0.0
        self._stats_at = now

    @staticmethod
    def _wait_until(deadline: float) -> None:
        """Sleep up to _SPIN_MARGIN before the deadline, then busy-spin the remainder."""
        coarse = deadline - perf_counter() - _SPIN_MARGIN
        if coarse > 0:
            sleep(coarse)
        while perf_counter() < deadline:
            pass
