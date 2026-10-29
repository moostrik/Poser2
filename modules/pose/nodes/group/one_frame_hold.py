"""OneFrameHold — one feature delayed by one frame.

A group step that stamps each pose with the feature it carried in the previous frame: the value the
step receives this frame is held and stamped next frame. A pose that was not in the previous frame
gets the feature's NaN dummy. It keeps a lag that a chain used to have, so the lag is one visible step
in the app's pipeline file and removing it is deleting that step.
"""

from __future__ import annotations

from ...features import BaseFeature
from ...frame import FrameDict, replace
from ..nodes import GroupNode


class OneFrameHold(GroupNode):
    """Stamps the previous frame's ``feature_type`` on every pose; see the module docstring."""

    def __init__(self, feature_type: type[BaseFeature]) -> None:
        self._feature_type = feature_type
        self._previous: FrameDict = {}

    @property
    def feature_type(self) -> type[BaseFeature]:
        return self._feature_type

    def process(self, frames: FrameDict) -> FrameDict:
        ft = self._feature_type
        previous = self._previous
        out: FrameDict = {}
        for tid, frame in frames.items():
            held = previous[tid][ft] if tid in previous else ft.create_dummy()
            out[tid] = replace(frame, {ft: held})
        self._previous = frames
        return out

    def reset(self) -> None:
        self._previous = {}
