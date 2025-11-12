"""Base class for pose filter trackers."""

from abc import ABC, abstractmethod

from modules.pose.Pose import Pose, PoseDict
from modules.pose.callback import PoseDictCallbackMixin


class FilterTrackerBase(PoseDictCallbackMixin, ABC):
    """Base class for tracking and filtering multiple poses.

    Provides callback system and reset functionality. Subclasses implement
    the specific filtering logic.
    """

    def __init__(self, num_poses: int):
        """Initialize base tracker.

        Args:
            num_poses: Number of poses to track.
        """
        super().__init__()  # Initialize mixin
        self.num_poses: int = num_poses

    def add_poses(self, poses: PoseDict) -> None:
        """Process poses and emit callbacks.

        Args:
            poses: Dictionary of poses to process.
        """
        pending_poses: dict[int, Pose] = {}

        for pose_id, pose in poses.items():
            try:
                current_pose: Pose = self._process_pose(pose_id, pose)
                pending_poses[pose_id] = current_pose
            except Exception as e:
                print(f"{self.__class__.__name__}: Error processing pose {pose_id}: {e}")
                from traceback import print_exc
                print_exc()
                pending_poses[pose_id] = pose

            # Reset when pose is lost
            if pose.lost:
                self.reset_pose(pose_id)

        self._notify_callbacks(pending_poses)

    @abstractmethod
    def _process_pose(self, pose_id: int, pose: Pose) -> Pose:
        """Process a single pose through filters.

        Args:
            pose_id: ID of the pose.
            pose: Pose to process.

        Returns:
            Processed pose.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all pose filters."""
        pass

    @abstractmethod
    def reset_pose(self, pose_id: int) -> None:
        """Reset filter for a specific pose.

        Args:
            pose_id: ID of pose to reset.
        """
        pass