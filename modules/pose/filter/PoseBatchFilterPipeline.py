# Standard library imports
from threading import Lock
from traceback import print_exc
from typing import Callable

# Pose imports
from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.filter.PoseFilterBase import PoseFilterBase

from modules.Settings import Settings


class PoseBatchFilterPipeline:
    """Manages pose processing through filter chains.

    Maintains a fixed number of filter chains (one per pose ID).
    Each chain processes poses sequentially through multiple filters.
    """

    def __init__(self, num_poses: int, filters: list[Callable[[], PoseFilterBase]]) -> None:
        if not filters:
            raise ValueError("PosePipeline: filter_factories must not be empty.")

        self.num_poses: int = num_poses
        self._output_callbacks: set[PoseDictCallback] = set()
        self._callback_lock = Lock()

        # Create fixed dict of filter chains (one per pose ID)
        self._filter_chains: dict[int, list[PoseFilterBase]] = {}
        for pose_id in range(self.num_poses):
            chain: list[PoseFilterBase] = [filter() for filter in filters]
            self._filter_chains[pose_id] = chain

    def add_poses(self, poses: PoseDict) -> None:
        """Process incoming pose batch through filter chains."""
        pending_poses: dict[int, Pose] = {}
        for pose_id, pose in poses.items():

            chain: list[PoseFilterBase] = self._filter_chains[pose_id]

            try:
                current_pose: Pose = pose
                for filter_instance in chain:
                    current_pose = filter_instance.process(current_pose)
                pending_poses[pose_id] = current_pose
            except Exception as e:
                print(f"PosePipeline: Error processing pose {pose_id}: {e}")
                print_exc()
                pending_poses[pose_id] = pose

            if pose.lost:
                self.reset_pose(pose_id)

        self._emit_callbacks(pending_poses)

    def reset(self) -> None:
        for chain in self._filter_chains.values():
            for filter_instance in chain:
                filter_instance.reset()

    def reset_pose(self, pose_id: int) -> None:
        chain: list[PoseFilterBase] = self._filter_chains[pose_id]
        for filter_instance in chain:
            filter_instance.reset()

    # CALLBACKS
    def _emit_callbacks(self, poses: dict[int, Pose]) -> None:
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"PosePipeline: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_callback(self, callback: PoseDictCallback) -> None:
        with self._callback_lock:
            self._output_callbacks.discard(callback)