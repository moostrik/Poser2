# Standard library imports
from dataclasses import replace
from threading import Lock

from modules.pose.features.LeaderScore import LeaderScore, configure_leader_score
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame


class LeaderScoreApplicator(FilterNode):
    """Filter that applies pre-computed leader score data to poses.

    Leader scores track temporal offset between pose pairs:
    - Negative value: this pose leads the other
    - Positive value: other pose leads this one
    - Zero: synchronized movement

    Thread-safe: Uses lock to protect stored leader score dict.
    """

    def __init__(self, max_poses: int) -> None:
        configure_leader_score(max_poses)
        self._leader_dict: dict[int, LeaderScore] = {}
        self._lock: Lock = Lock()

    def submit(self, leader_dict: dict[int, LeaderScore]) -> None:
        """Store the per-pose leader score dict for processing.

        Args:
            leader_dict: Maps track_id -> LeaderScore object
        """
        with self._lock:
            self._leader_dict = leader_dict

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
            pose = replace(pose, leader=leader)

        return pose
