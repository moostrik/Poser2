"""Tracks and processes multiple poses through filter pipelines.

Maintains separate filter pipelines for each pose ID, automatically
resetting pipelines when poses are lost.
"""

from threading import Lock
from traceback import print_exc
from typing import Callable

from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.filter.PoseFilterBase import PoseFilterBase


class PoseFilterPipelineTracker:
    """Tracks multiple poses, maintaining a separate filter pipeline for each.

    Each pose_id gets its own chain of filters which maintains independent state.
    Pipelines are automatically reset when their pose is lost.
    """

    def __init__(self, num_poses: int, filter_factories: list[Callable[[], PoseFilterBase]]) -> None:
        if not filter_factories:
            raise ValueError("PoseFilterPipelineTracker: filter_factories must not be empty.")

        self.num_poses: int = num_poses
        self._output_callbacks: set[PoseDictCallback] = set()
        self._callback_lock = Lock()

        # Create one filter pipeline per pose ID
        self._filter_pipelines: dict[int, list[PoseFilterBase]] = {}
        for pose_id in range(self.num_poses):
            pipeline: list[PoseFilterBase] = [factory() for factory in filter_factories]
            self._filter_pipelines[pose_id] = pipeline

    def add_poses(self, poses: PoseDict) -> None:
        """Process poses through their individual filter pipelines."""
        pending_poses: dict[int, Pose] = {}

        for pose_id, pose in poses.items():
            pipeline: list[PoseFilterBase] = self._filter_pipelines[pose_id]

            try:
                current_pose: Pose = pose
                for filter_instance in pipeline:
                    current_pose = filter_instance.process(current_pose)
                pending_poses[pose_id] = current_pose
            except Exception as e:
                print(f"PoseFilterPipelineTracker: Error processing pose {pose_id}: {e}")
                print_exc()
                pending_poses[pose_id] = pose

            # Reset pipeline when pose is lost
            if pose.lost:
                self.reset_pose(pose_id)

        self._emit_callbacks(pending_poses)

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

    # CALLBACKS
    def _emit_callbacks(self, poses: dict[int, Pose]) -> None:
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"PoseFilterPipelineTracker: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.discard(callback)