"""Tracks and interpolates multiple poses independently."""

from time import perf_counter

from modules.utils import PerformanceTimer

from ..frame import FrameDict
from .interpolator_pipeline import InterpolatorPipeline
from .tracker_base import TrackerBase

import logging
logger = logging.getLogger(__name__)


class InterpolatorTracker(TrackerBase):
    """Multiplexes a FrameDict across per-track InterpolatorPipelines.

    Handles dispatch, result collection, lifecycle reset on track
    disappearance, and callback fan-out. Pipeline construction is
    the caller's responsibility. A `name` times every update (the pipelines
    only, not the callbacks) for the timing session.
    """

    def __init__(self, pipelines: dict[int, InterpolatorPipeline], name: str | None = None) -> None:
        super().__init__()
        if not pipelines:
            raise ValueError("pipelines dict must not be empty.")
        self._pipelines = pipelines
        self._timer: PerformanceTimer | None = PerformanceTimer(name=name, sample_count=200) if name else None

    def set(self, poses: FrameDict) -> None:
        """Set target poses for interpolation."""

        # Reset pipelines for tracks that are no longer present
        for id in self._pipelines:
            if id not in poses:
                self.reset_at(id)

        for id, pose in poses.items():
            try:
                self._pipelines[id].set(pose)
            except Exception as e:
                logger.error(f"Error setting pose {id}: {e}")
                # Some nodes may hold the new target and others not; don't emit a half-updated frame
                self.reset_at(id)

    def update(self) -> FrameDict:
        """Get interpolated poses from all pipelines."""

        start = perf_counter()
        interpolated_poses: FrameDict = {}

        for id, pipeline in self._pipelines.items():
            try:
                pose = pipeline.update()
            except Exception as e:
                logger.error(f"Error updating pose {id}: {e}")
                continue
            if pose is not None:
                interpolated_poses[id] = pose

        if self._timer is not None:
            self._timer.add_time((perf_counter() - start) * 1000.0, report=False)

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