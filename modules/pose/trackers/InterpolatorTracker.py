"""Tracks and interpolates multiple poses independently."""

from modules.pose.frame import FrameDict
from .InterpolatorPipeline import InterpolatorPipeline
from .TrackerBase import TrackerBase

import logging
logger = logging.getLogger(__name__)


class InterpolatorTracker(TrackerBase):
    """Multiplexes a FrameDict across per-track InterpolatorPipelines.

    Handles dispatch, result collection, lifecycle reset on track
    disappearance, and callback fan-out. Pipeline construction is
    the caller's responsibility.
    """

    def __init__(self, pipelines: dict[int, InterpolatorPipeline]) -> None:
        super().__init__()
        if not pipelines:
            raise ValueError("InterpolatorTracker: pipelines dict must not be empty.")
        self._pipelines = pipelines

    def submit(self, poses: FrameDict) -> None:
        """Submit target poses for interpolation."""

        # Reset pipelines for tracks that are no longer present
        for id in self._pipelines:
            if id not in poses:
                self.reset_at(id)

        try:
            for id, pose in poses.items():
                self._pipelines[id].submit(pose)
        except Exception as e:
            logger.error(f"InterpolatorTracker: Error submitting pose {id}: {e}")

    def update(self) -> FrameDict:
        """Get interpolated poses from all pipelines."""

        interpolated_poses: FrameDict = {}

        try:
            for id, pipeline in self._pipelines.items():
                pose = pipeline.update()
                if pose is not None:
                    interpolated_poses[id] = pose
        except Exception as e:
            logger.error(f"InterpolatorTracker: Error updating pose {id}: {e}")

        self._notify_frames_callbacks(interpolated_poses)

        return interpolated_poses

    def reset(self) -> None:
        """Reset all pipelines."""
        for pipeline in self._pipelines.values():
            pipeline.reset()

    def reset_at(self, id: int) -> None:
        """Reset pipeline for a specific track."""
        if id in self._pipelines:
            self._pipelines[id].reset()