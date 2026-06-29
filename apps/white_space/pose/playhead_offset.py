"""PlayheadPhase — a White Space-native pose feature: the signed phase offset of a
pose's azimuth relative to the rotating light playhead.

The playhead is a White Space concept, so the feature and its extractor live with the
app rather than in ``modules/pose``. The open Frame ECS still lets the feature ride on
``Frame`` (via ``replace``) without modules depending on app code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from modules.pose.features import Azimuth, SingleAngle
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode


class PlayheadOffset(SingleAngle):
    """Signed phase offset in radians [-π, π) of a pose's azimuth relative to the playhead.

    ``0`` = playhead on the pose, ``>0`` = playhead approaching (pose ahead in the sweep
    direction), ``<0`` = playhead just passed, ``±π`` = opposite side of the ring.
    Absent (NaN, score 0.0) when no playhead or azimuth is available.
    """


class PlayheadOffsetExtractor(FilterNode):
    """Stamps ``PlayheadPhase`` from the pose azimuth and a live playhead provider.

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
