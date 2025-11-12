"""Tracks and filters multiple poses independently."""

from typing import Callable

from modules.pose.Pose import Pose
from modules.pose.nodes.Nodes import FilterNode
from .TrackerBase import TrackerBase


class FilterTracker(TrackerBase):
    """Tracks multiple poses, maintaining a separate filter for each.

    Each pose_id gets its own PoseFilterBase instance which maintains
    independent state. Filters are automatically reset when their pose is lost.
    """

    def __init__(self, num_poses: int, filter_factory: Callable[[], FilterNode]) -> None:
        """Initialize tracker with single filter per pose.

        Args:
            num_poses: Number of poses to track.
            filter_factory: Factory function that creates filter instances.
        """
        super().__init__(num_poses)

        # Create one filter instance per pose ID
        self._filters: dict[int, FilterNode] = {
            pose_id: filter_factory() for pose_id in range(num_poses)
        }

    def _process_pose(self, pose_id: int, pose: Pose) -> Pose:
        """Process pose through its filter."""
        return self._filters[pose_id].process(pose)

    def reset(self) -> None:
        """Reset all pose filters."""
        for filter_instance in self._filters.values():
            filter_instance.reset()

    def reset_pose(self, pose_id: int) -> None:
        """Reset filter for a specific pose."""
        self._filters[pose_id].reset()