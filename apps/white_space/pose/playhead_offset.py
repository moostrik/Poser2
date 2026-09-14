"""PlayheadOffset — a White Space-native pose feature: the signed angular offset of a
pose's azimuth relative to the rotating light playhead.

The playhead is a White Space concept, so the feature and its extractor live with the
app rather than in ``modules/pose``. The open Frame ECS still lets the feature ride on
``Frame`` (via ``replace``) without modules depending on app code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from modules.pose.features import Azimuth, SingleAngle
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode

# Only the near half of the sweep has a crossing; the far side (|offset| → π) never does.
_HALF_PI: float = math.pi / 2.0


def playhead_step(beam_rpm: float, tick_interval: float) -> float:
    """The playhead's advance per light tick (radians) at the content-sweep rate."""
    return beam_rpm / 60.0 * math.tau * tick_interval


def ticks_to_crossing(offset: float, step: float) -> float:
    """Ticks until the playhead crosses the pose (``PlayheadOffset`` 0): ``offset / step`` —
    positive approaching, negative just past. ``step`` is ``playhead_step``, the same for every
    caller, so the flash and the state machine's hit pick the same tick.

    The current tick is among the N ticks closest to the crossing when ``|τ| < N / 2``: N = 1 is the
    closest tick, 2 the pair straddling it, 3 the closest and both neighbours. Needs no history, so
    a layer that was not drawn last tick answers the same. NaN when the offset is NaN, on the far half
    of the sweep (``|offset| ≥ π/2``), or when the playhead does not move (``step ≤ 0``).
    """
    if math.isnan(offset) or abs(offset) >= _HALF_PI or step <= 0.0:
        return math.nan
    return offset / step


@dataclass(frozen=True, slots=True)
class _Pass:
    """One pose's state through the current pass: the last offset and the ticks already lit."""
    offset: float
    fired:  int


class PlayheadCrossing:
    """Which poses the playhead crosses this tick: the ``frames`` ticks closest to each crossing.

    A tick lights when it is among the ``frames`` closest (``ticks_to_crossing``). A per-pose count
    stops a jittered step (a playhead tracking correction, a person moving) from lighting an extra
    tick, and a sign flip + → − with nothing lit yet fires once, so a pass is never skipped. The
    count starts over once the playhead is a quarter turn away. The closest-tick measure needs no
    history, so the first tick after ``reset`` still lights.
    """

    def __init__(self) -> None:
        self._passes: dict[int, _Pass] = {}

    def reset(self) -> None:
        self._passes.clear()

    def update(self, offsets: dict[int, float], step: float, frames: int) -> set[int]:
        """Advance every pose's pass by this tick's ``PlayheadOffset`` (per track id) and return the
        ids lit this tick. ``step`` is ``playhead_step``; ids absent from ``offsets`` are forgotten."""
        lit: set[int] = set()
        passes: dict[int, _Pass] = {}
        for id, offset in offsets.items():
            last = self._passes.get(id)
            prev = last.offset if last is not None else math.nan
            fired = last.fired if last is not None else 0
            if math.isnan(offset) or abs(offset) >= _HALF_PI:
                fired = 0                                        # no pass under way
            else:
                tau = ticks_to_crossing(offset, step)
                closest = not math.isnan(tau) and abs(tau) < frames / 2.0
                stepped_over = fired == 0 and prev > 0.0 >= offset     # jitter skipped every close tick
                if fired < frames and (closest or stepped_over):
                    fired += 1
                    lit.add(id)
            passes[id] = _Pass(offset, fired)
        self._passes = passes
        return lit


class PlayheadOffset(SingleAngle):
    """Signed angular offset in radians [-π, π) of a pose's azimuth relative to the playhead.

    ``0`` = playhead on the pose, ``>0`` = playhead approaching (pose ahead in the sweep
    direction), ``<0`` = playhead just passed, ``±π`` = opposite side of the circle.
    Absent (NaN, score 0.0) when no playhead or azimuth is available.
    """


class PlayheadOffsetExtractor(FilterNode):
    """Stamps ``PlayheadOffset`` from the pose azimuth and a live playhead provider.

    ``playhead`` returns the current playhead (radians [-π, π); NaN when not meaningful).
    Wired in main to ``board.get_playhead`` — read live so there is no lag.
    """

    def __init__(self, playhead: Callable[[], float]) -> None:
        self._playhead = playhead

    def process(self, pose: Frame) -> Frame:
        azimuth: float = pose[Azimuth].value
        playhead: float = self._playhead()
        if np.isnan(azimuth) or np.isnan(playhead):
            return pose

        # SingleAngle.from_value wraps the difference to the signed shortest angle [-π, π).
        return replace(pose, {PlayheadOffset: PlayheadOffset.from_value(azimuth - playhead)})
