"""Tracks and filters multiple poses independently.

Maintains separate filter instances for each pose ID, automatically
resetting filters when poses are lost.
"""

from threading import Lock
from traceback import print_exc
from typing import Callable

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.filter.PoseFilterBase import PoseFilterBase


class PoseFilterTracker:
    """Tracks multiple poses, maintaining a separate filter for each.

    Each pose_id gets its own PoseFilterBase instance which maintains
    independent state. Filters are automatically reset when their pose is lost.
    """

    def __init__(self, num_poses: int, filter_factory: Callable[[], PoseFilterBase]) -> None:
        self.num_poses: int = num_poses
        self._output_callbacks: set[PoseDictCallback] = set()
        self._callback_lock = Lock()

        # Create one filter instance per pose ID
        self._filters: dict[int, PoseFilterBase] = {
            pose_id: filter_factory() for pose_id in range(num_poses)
        }

    def add_poses(self, poses: PoseDict) -> None:
        """Process poses through their individual filters."""
        pending_poses: dict[int, Pose] = {}

        for pose_id, pose in poses.items():
            filter_instance: PoseFilterBase = self._filters[pose_id]

            try:
                current_pose: Pose = filter_instance.process(pose)
                pending_poses[pose_id] = current_pose
            except Exception as e:
                print(f"PoseFilterTracker: Error processing pose {pose_id}: {e}")
                print_exc()
                pending_poses[pose_id] = pose

            # Reset filter when pose is lost
            if pose.lost:
                self.reset_pose(pose_id)

        self._emit_callbacks(pending_poses)

    def reset(self) -> None:
        """Reset all pose filters."""
        for filter_instance in self._filters.values():
            filter_instance.reset()

    def reset_pose(self, pose_id: int) -> None:
        """Reset filter for a specific pose."""
        self._filters[pose_id].reset()

    # CALLBACKS
    def _emit_callbacks(self, poses: dict[int, Pose]) -> None:
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"PoseFilterTracker: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.discard(callback)