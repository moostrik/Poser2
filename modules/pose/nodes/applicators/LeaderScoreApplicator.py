# Standard library imports
from threading import Lock

from modules.pose.features.LeaderScore import LeaderScore
from modules.pose.batch.WindowSimilarity import SimilarityResult
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.frame import Frame, replace
from modules.settings import BaseSettings, Field


class LeaderScoreApplicatorSettings(BaseSettings):
    """Configuration for LeaderScoreApplicator."""
    max_poses: Field[int] = Field(4, access=Field.INIT, min=1, max=16)


class LeaderScoreApplicator(FilterNode):
    """Filter that applies pre-computed leader score data to poses.

    Leader scores track temporal offset between pose pairs:
    - Negative value: this pose leads the other
    - Positive value: other pose leads this one
    - Zero: synchronized movement

    Thread-safe: Uses lock to protect stored leader score dict.
    """

    def __init__(self, settings: LeaderScoreApplicatorSettings | None = None) -> None:
        self._settings = settings if settings is not None else LeaderScoreApplicatorSettings()
        self._leader_dict: dict[int, LeaderScore] = {}
        self._lock: Lock = Lock()

    def set(self, leader_dict: dict[int, LeaderScore]) -> None:
        """Store the per-pose leader score dict for processing.

        Args:
            leader_dict: Maps track_id -> LeaderScore object
        """
        with self._lock:
            self._leader_dict = leader_dict

    def set_result(self, result: SimilarityResult) -> None:
        self.set(result.leader_score)

    def process(self, pose: Frame) -> Frame:
        """Apply pre-computed leader scores to this pose.

        Args:
            pose: Frame to enrich with leader score data

        Returns:
            Frame with updated leader field
        """
        with self._lock:
            leader: LeaderScore | None = self._leader_dict.get(pose.track_id)

        if leader is not None:
            pose = replace(pose, {LeaderScore: leader})

        return pose
