"""PostureSimilarityExtractor — stamps every pose with its posture similarity to the others of the frame.

A group step: the pairwise posture similarity of the frame's poses (``analytics.posture_similarity_rows``:
0..1 per pair, score = coverage), each pose's row stamped as its ``Similarity``. A pose alone reads NaN
toward everyone; a pose whose id is past the row's width gets no row. Synchronous, on the frame's thread.
"""

from __future__ import annotations

from ...analytics import PostureSimilaritySettings, posture_similarity_rows
from ...frame import FrameDict, replace
from ...features import Similarity
from ..nodes import GroupNode


class PostureSimilarityExtractor(GroupNode):
    """Stamps the posture similarity rows of the frame's poses; see the module docstring."""

    def __init__(self, settings: PostureSimilaritySettings | None = None) -> None:
        self.settings = settings if settings is not None else PostureSimilaritySettings()

    def process(self, frames: FrameDict) -> FrameDict:
        rows = posture_similarity_rows(frames, self.settings)
        return {id: replace(frame, {Similarity: rows[id]}) if id in rows else frame
                for id, frame in frames.items()}
