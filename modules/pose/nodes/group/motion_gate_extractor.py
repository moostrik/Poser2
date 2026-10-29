"""MotionGateExtractor — stamps every pose with the motion gate between it and the others of the frame.

A group step: ``gate[A, B] = motion_A * motion_B``, where a pose's motion is its ``AngleMotion`` value (0 when
NaN), and ``gate[A, A]`` the pose's own motion. The row is as wide as the configured track features; a pose
whose id is past that width still gets a row of its gates toward the others. Synchronous, on the frame's thread.
"""

from __future__ import annotations

import numpy as np

from ...features import AngleMotion, MotionGate
from ...frame import FrameDict, replace
from ..nodes import GroupNode


class MotionGateExtractor(GroupNode):
    """Stamps the motion gate rows of the frame's poses; see the module docstring."""

    def process(self, frames: FrameDict) -> FrameDict:
        length = MotionGate.length()
        motions: dict[int, float] = {}
        for tid, frame in frames.items():
            motion = frame[AngleMotion].value
            motions[tid] = motion if not np.isnan(motion) else 0.0

        out: FrameDict = {}
        for tid, frame in frames.items():
            values = np.zeros(length, dtype=np.float32)
            scores = np.zeros(length, dtype=np.float32)
            own = motions[tid]
            if tid < length:
                values[tid] = own
                scores[tid] = 1.0
            for other, motion in motions.items():
                if other != tid and other < length:
                    values[other] = own * motion
                    scores[other] = 1.0
            out[tid] = replace(frame, {MotionGate: MotionGate(values=values, scores=scores)})
        return out
