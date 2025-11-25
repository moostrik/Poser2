"""Tracks and filters multiple poses independently."""

from traceback import print_exc
from typing import Callable

from .TrackerBase import TrackerBase
from modules.pose.nodes.Nodes import FilterNode
from modules.pose.Frame import Frame, FrameDict


class FilterTracker(TrackerBase):
    """Tracks multiple poses, maintaining a separate filter or filter pipeline for each.

    Each pose_id gets its own FilterNode or chain of FilterNodes which maintains
    independent state. Filters are automatically reset when their pose is lost.
    """

    def __init__(self, num_tracks: int, filter_factory: Callable[[], FilterNode] | list[Callable[[], FilterNode]]) -> None:
        """Initialize tracker with filter(s) per pose."""
        super().__init__()  # Initialize PoseDictCallbackMixin

        # Convert single factory to list for uniform handling
        if callable(filter_factory):
            self._filter_factories = [filter_factory]
        else:
            if not filter_factory:
                raise ValueError("FilterTracker: filter_factory list must not be empty.")
            self._filter_factories = filter_factory

        self._filter_pipelines: dict[int, list[FilterNode]] = {
            id: [factory() for factory in self._filter_factories]
            for id in range(num_tracks)
        }

    def process(self, poses: FrameDict) -> FrameDict:
        """Process poses through filters and emit callbacks."""

        # Reset interpolators for poses that are no longer present
        for id in self._filter_pipelines:
            if id not in poses:
                self.reset_at(id)

        filtered_poses: FrameDict = {}

        for id, pose in poses.items():
            try:
                filtered_pose: Frame = pose
                for filter_node in self._filter_pipelines[id]:
                    filtered_pose = filter_node.process(filtered_pose)
                filtered_poses[id] = filtered_pose
            except Exception as e:
                print(f"FilterTracker: Error processing pose {id}: {e}")
                print_exc()
                filtered_poses[id] = pose

        self._notify_poses_callbacks(filtered_poses)

        return filtered_poses

    def reset(self) -> None:
        """Reset all pose filter pipelines."""
        for pipeline in self._filter_pipelines.values():
            for filter_node in pipeline:
                filter_node.reset()

    def reset_at(self, id: int) -> None:
        """Reset filters in pipeline for a specific pose."""
        for filter_node in self._filter_pipelines[id]:
            filter_node.reset()