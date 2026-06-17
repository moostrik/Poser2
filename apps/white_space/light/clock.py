"""Master clock — produces a per-tick time snapshot (`Tick`)."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import LightRendererSettings


@dataclass
class Tick:
    """Immutable clock snapshot produced once per render tick."""
    time:  float   # absolute elapsed seconds since clock start
    dt:    float   # seconds elapsed since the previous tick
    bpm:   float   # current master tempo in beats per minute
    phase: float   # beat phase: 0.0 = beat start, approaching 1.0 = next beat
    beat:  int     # monotonic beat counter (increments each time phase wraps)


class Clock:
    """Advances the master clock once per render tick."""

    def __init__(self, settings: LightRendererSettings) -> None:
        self._settings   = settings
        self._start_time: float = time()
        self._last_time:  float = self._start_time
        self._phase_acc: float = 0.0
        self._beat:      int   = 0

    def tick(self) -> Tick:
        now     = time()
        dt      = now - self._last_time
        elapsed = now - self._start_time
        self._last_time = now

        bpm = self._settings.bpm
        self._phase_acc += dt * bpm / 60.0
        if self._phase_acc >= 1.0:
            full_beats       = int(self._phase_acc)
            self._beat      += full_beats
            self._phase_acc -= full_beats

        t = Tick(
            time  = elapsed,
            dt    = dt,
            bpm   = bpm,
            phase = self._phase_acc,
            beat  = self._beat,
        )
        self._update_readouts(t)
        return t

    def _update_readouts(self, t: Tick) -> None:
        self._settings.time  = t.time
        self._settings.phase = t.phase
