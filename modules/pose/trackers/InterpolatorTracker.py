"""Tracks and interpolates multiple poses independently."""

from traceback import print_exc
from typing import Callable

from modules.pose.Pose import Pose, PoseDict
from modules.pose.nodes.Nodes import InterpolatorNode
from .TrackerBase import TrackerBase


class InterpolatorTracker(TrackerBase):
    """Tracks multiple poses, maintaining a separate interpolator for each."""

    def __init__(self, num_tracks: int, interpolator_factory: Callable[[], InterpolatorNode]) -> None:
        """Initialize tracker with interpolators for fixed number of poses."""
        super().__init__() # Initialize PoseDictCallbackMixin

        self._interpolators: dict[int, InterpolatorNode] = {
            id: interpolator_factory()
            for id in range(num_tracks)
        }

    def submit(self, poses: PoseDict) -> None:
        """Submit target poses for interpolation."""

        # Reset interpolators for poses that are no longer present
        for id in self._interpolators:
            if id not in poses:
                self.reset_at(id)

        # Submit poses that are present
        for id, pose in poses.items():
            try:
                self._interpolators[id].submit(pose)
            except Exception as e:
                print(f"InterpolatorTracker: Error submitting pose {id}: {e}")
                print_exc()

    def update(self, current_time: float | None = None) -> PoseDict:
        """Get interpolated poses at current time."""

        interpolated_poses: PoseDict = {}

        for id, interpolator in self._interpolators.items():
            try:
                pose: Pose | None = interpolator.update(current_time)
                if pose is not None:
                    interpolated_poses[id] = pose

            except Exception as e:
                print(f"InterpolatorTracker: Error updating pose {id}: {e}")
                from traceback import print_exc
                print_exc()

        # Emit callbacks with interpolated poses
        if interpolated_poses:
            self._notify_pose_dict_callbacks(interpolated_poses)

        return interpolated_poses

    def reset(self) -> None:
        """Reset all interpolators."""
        for interpolator in self._interpolators.values():
            interpolator.reset()

    def reset_at(self, id: int) -> None:
        """Reset interpolator for a specific pose ID."""
        if id in self._interpolators:
            self._interpolators[id].reset()