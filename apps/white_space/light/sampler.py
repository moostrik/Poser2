"""Playhead sampler — detects when the rotating playhead sweeps past active players.

Owns its own cross-tick state (previous playhead position and last-tick wall time)
so the renderer does not have to track it.
"""

from dataclasses import dataclass
from time import monotonic

import numpy as np

from modules.tracker import Tracklet
from modules.pose.frame import Frame
from modules.pose import features

from .motor import MotorMode


@dataclass(frozen=True)
class Hit:
    """Emitted when the rotating playhead sweeps past an active player."""
    track_id: int
    position: float  # azimuth 0.0–1.0 of the person at moment of crossing


def sweep_contains(prev: float, curr: float, pos: float) -> bool:
    """Return True if *pos* falls inside (prev, curr] on the unit circle, with wrap-around."""
    if prev <= curr:
        return prev < pos <= curr
    else:
        return pos > prev or pos <= curr


class Sampler:
    """Detects playhead crossings; owns previous-playhead and gap-check state."""

    def __init__(self) -> None:
        self._prev_playhead:  float = 0.0
        self._last_tick_mono: float = float('-inf')

    def detect(
        self,
        frames:    list[Frame],
        tracklets: dict[int, Tracklet],
        playhead:  float,
        mode:      MotorMode,
    ) -> tuple[Hit, ...]:
        """Return one Hit per active player whose azimuth was crossed since the previous tick.

        Detection is skipped after a long gap (e.g. first tick or a stall) and when the
        motor is not in LOW_SPEED, where playhead tracking is not meaningful.
        """
        now_mono = monotonic()
        gap_ok   = (now_mono - self._last_tick_mono) < 0.5

        hits: tuple[Hit, ...] = ()
        if gap_ok and mode == MotorMode.LOW_SPEED:
            hits = self._crossings(frames, tracklets, self._prev_playhead, playhead)

        self._prev_playhead  = playhead
        self._last_tick_mono = now_mono
        return hits

    @staticmethod
    def _crossings(
        frames:    list[Frame],
        tracklets: dict[int, Tracklet],
        prev:      float,
        curr:      float,
    ) -> tuple[Hit, ...]:
        hits: list[Hit] = []
        for frame in frames:
            tracklet = tracklets.get(frame.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            pos: float = frame[features.Azimuth].value
            if not np.isnan(pos) and sweep_contains(prev, curr, pos):
                hits.append(Hit(track_id=frame.track_id, position=pos))
        return tuple(hits)
