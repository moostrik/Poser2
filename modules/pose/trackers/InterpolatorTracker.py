"""Tracks and interpolates multiple poses independently."""

from dataclasses import replace
from traceback import print_exc
from time import monotonic
from typing import Callable

from modules.pose.Pose import Pose, PoseDict
from modules.pose.nodes.Nodes import InterpolatorNode
from .TrackerBase import TrackerBase

# MAKE AVAILABLE FOR FACTORY LISTS (like filter trackers)
class InterpolatorTracker(TrackerBase):
    """Tracks multiple poses, maintaining a separate interpolator for each."""

    def __init__(self, num_tracks: int, interpolator_factory: Callable[[], InterpolatorNode] | list[Callable[[], InterpolatorNode]]) -> None:
        """Initialize tracker with interpolators for fixed number of poses."""
        super().__init__() # Initialize PoseDictCallbackMixin

        # Convert single factory to list for uniform handling
        if callable(interpolator_factory):
            _interpolator_factory = [interpolator_factory]
        else:
            if not interpolator_factory:
                raise ValueError("FilterTracker: filter_factory list must not be empty.")
            _interpolator_factory = interpolator_factory

        self._interpolator_pipeline: dict[int, list[InterpolatorNode]] = {
            id: [factory() for factory in _interpolator_factory]
            for id in range(num_tracks)
        }

    def submit(self, poses: PoseDict) -> None:
        """Submit target poses for interpolation."""

        # Reset interpolators for poses that are no longer present
        for id in self._interpolator_pipeline:
            if id not in poses:
                self.reset_at(id)

        # Submit poses that are present
        try:
            for id, pose in poses.items():
                for node in self._interpolator_pipeline[id]:
                    node.submit(pose)
        except Exception as e:
            print(f"InterpolatorTracker: Error submitting pose {id}: {e}")
            print_exc()

    def update(self, time_stamp: float | None = None) -> PoseDict:
        """Get interpolated poses at current time."""

        interpolated_poses: PoseDict = {}

        if time_stamp is None:
            time_stamp = monotonic()

        try:
            for id, pipeline in self._interpolator_pipeline.items():
                # If there are multiple interpolators per track, you may want to combine their outputs.
                # Here, we just use the first interpolator's output.
                pose: Pose | None = None
                for node in pipeline:
                    interpolated_pose: Pose | None = node.update(time_stamp)
                    attractor: str = node.attr_name
                    if interpolated_pose is not None:
                        if pose is None:
                            # Use first interpolator's pose as base
                            pose = interpolated_pose
                        else:
                            # Merge subsequent interpolator's feature into combined pose
                            pose = replace(pose, **{attractor: getattr(interpolated_pose, attractor)})

                if pose is not None:
                    interpolated_poses[id] = pose

        except Exception as e:
            print(f"InterpolatorTracker: Error updating pose {id}: {e}")
            from traceback import print_exc
            print_exc()

        # Emit callbacks with interpolated poses
        self._notify_poses_callbacks(interpolated_poses)

        return interpolated_poses

    def reset(self) -> None:
        """Reset all interpolators."""
        for pipeline in self._interpolator_pipeline.values():
            for node in pipeline:
                node.reset()

    def reset_at(self, id: int) -> None:
        """Reset interpolator for a specific pose ID."""
        if id in self._interpolator_pipeline:
            for node in self._interpolator_pipeline[id]:
                node.reset()