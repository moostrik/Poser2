"""SimilarityResultHold — the hold crossing from a threaded analytic back into the frame domain.

``WindowSimilarity`` and ``WindowCorrelation`` compute on their own thread and hand their
``SimilarityResult`` to ``set``; every frame then reads the latest result without waiting and stamps
each pose's row as its ``Similarity`` and ``LeaderScore``. A pose without a row keeps its frame as it is
(no data reads as NaN). A new result replaces the held one whole, so a result without rows clears them.
"""

from __future__ import annotations

from threading import Lock

from ...analytics import SimilarityResult
from ...features import LeaderScore, Similarity
from ...frame import FrameDict, replace
from ..nodes import GroupNode


class SimilarityResultHold(GroupNode):
    """Holds the latest ``SimilarityResult`` and stamps it on every frame; see the module docstring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._similarity: dict[int, Similarity] = {}
        self._leader_score: dict[int, LeaderScore] = {}

    def set(self, result: SimilarityResult) -> None:
        """Store the analytic's latest result (its thread)."""
        with self._lock:
            self._similarity = result.similarity
            self._leader_score = result.leader_score

    def process(self, frames: FrameDict) -> FrameDict:
        with self._lock:
            similarity = self._similarity
            leader_score = self._leader_score
        out: FrameDict = {}
        for tid, frame in frames.items():
            updates: dict = {}
            row = similarity.get(tid)
            if row is not None:
                updates[Similarity] = row
            lead = leader_score.get(tid)
            if lead is not None:
                updates[LeaderScore] = lead
            out[tid] = replace(frame, updates) if updates else frame
        return out

    def reset(self) -> None:
        with self._lock:
            self._similarity = {}
            self._leader_score = {}
