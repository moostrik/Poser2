"""Tracks and processes multiple poses through filter pipelines."""

from typing import Callable

from .PoseTrackerBase import PoseTrackerBase
from modules.pose.Pose import Pose
from modules.pose.filters.PoseFilterBase import PoseFilterBase


class PoseFilterPipelineTracker(PoseTrackerBase):
    """Tracks multiple poses, maintaining a separate filter pipeline for each.

    Each pose_id gets its own chain of filters which maintains independent state.
    Pipelines are automatically reset when their pose is lost.
    """

    def __init__(self, num_poses: int, filter_factories: list[Callable[[], PoseFilterBase]]) -> None:
        """Initialize tracker with filter pipeline per pose.

        Args:
            num_poses: Number of poses to track.
            filter_factories: List of factory functions that create filter instances.
        """
        if not filter_factories:
            raise ValueError("PoseFilterPipelineTracker: filter_factories must not be empty.")

        super().__init__(num_poses)

        # Create one filter pipeline per pose ID
        self._filter_pipelines: dict[int, list[PoseFilterBase]] = {}
        for pose_id in range(self.num_poses):
            pipeline: list[PoseFilterBase] = [factory() for factory in filter_factories]
            self._filter_pipelines[pose_id] = pipeline

    def _process_pose(self, pose_id: int, pose: Pose) -> Pose:
        """Process pose through its filter pipeline."""
        pipeline: list[PoseFilterBase] = self._filter_pipelines[pose_id]

        current_pose: Pose = pose
        for filter_instance in pipeline:
            current_pose = filter_instance.process(current_pose)

        return current_pose

    def reset(self) -> None:
        """Reset all pose filter pipelines."""
        for pipeline in self._filter_pipelines.values():
            for filter_instance in pipeline:
                filter_instance.reset()

    def reset_pose(self, pose_id: int) -> None:
        """Reset filter pipeline for a specific pose."""
        pipeline: list[PoseFilterBase] = self._filter_pipelines[pose_id]
        for filter_instance in pipeline:
            filter_instance.reset()