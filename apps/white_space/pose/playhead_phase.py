"""PlayheadPhase — a White Space-native pose feature: the signed phase offset of a
pose's azimuth relative to the rotating light playhead.

The playhead is a White Space concept, so the feature and its extractor live with the
app rather than in ``modules/pose``. The open Frame ECS still lets the feature ride on
``Frame`` (via ``replace``) without modules depending on app code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from modules.pose.features import Azimuth, SingleValue
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode


class PlayheadPhase(SingleValue):
    """Signed phase offset in [-1, 1] of a pose's azimuth relative to the light playhead.

    ``0`` = playhead on the pose, ``>0`` = playhead approaching (pose ahead in the sweep
    direction), ``<0`` = playhead just passed, ``±1`` = opposite side of the ring.
    Absent (NaN, score 0.0) when no playhead or azimuth is available.
    """

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (-1.0, 1.0)


class PlayheadPhaseExtractor(FilterNode):
    """Stamps ``PlayheadPhase`` from the pose azimuth and a live playhead provider.

    ``playhead`` returns the current playhead (revolution fraction [0,1); NaN when not
    meaningful). Wired in main to ``board.get_playhead`` — read live so there is no lag.
    """

    def __init__(self, playhead: Callable[[], float]) -> None:
        self._playhead = playhead

    def process(self, pose: Frame) -> Frame:
        azimuth: float = pose[Azimuth].value
        playhead: float = self._playhead()
        if np.isnan(azimuth) or np.isnan(playhead):
            return pose

        delta: float = (azimuth - (playhead % 1.0)) % 1.0   # 0..1, azimuth lead over playhead
        signed: float = delta - 1.0 if delta > 0.5 else delta  # -0.5..0.5
        value: float = signed * 2.0                            # -1..1 (÷ max offset)
        return replace(pose, {PlayheadPhase: PlayheadPhase.from_value(value)})
