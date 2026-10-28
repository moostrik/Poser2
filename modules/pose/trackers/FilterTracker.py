"""Tracks and filters multiple poses independently."""

from time import perf_counter

from modules.utils import PerformanceTimer

from .TrackerBase import TrackerBase
from .FilterPipeline import FilterPipeline
from ..frame import Frame, FrameDict

import logging
logger = logging.getLogger(__name__)


class FilterTracker(TrackerBase):
    """Multiplexes a FrameDict across per-track FilterPipelines.

    Handles dispatch, result collection, lifecycle reset on track
    disappearance, and callback fan-out. Pipeline construction is
    the caller's responsibility. A `name` times every call (the pipelines
    only, not the callbacks) for the timing session.
    """

    def __init__(self, pipelines: dict[int, FilterPipeline], name: str | None = None) -> None:
        super().__init__()
        if not pipelines:
            raise ValueError("pipelines dict must not be empty.")
        self._pipelines = pipelines
        self._timer: PerformanceTimer | None = PerformanceTimer(name=name, sample_count=200) if name else None

    def process(self, poses: FrameDict) -> FrameDict:
        """Process poses through per-track pipelines and emit callbacks."""

        start = perf_counter()

        # Reset pipelines for tracks that are no longer present
        for id in self._pipelines:
            if id not in poses:
                self.reset_at(id)

        filtered_poses: FrameDict = {}

        for id, pose in poses.items():
            try:
                filtered_poses[id] = self._pipelines[id].process(pose)
            except Exception as e:
                logger.error(f"Error processing pose {id}: {e}")
                filtered_poses[id] = pose

        if self._timer is not None:
            self._timer.add_time((perf_counter() - start) * 1000.0, report=False)

        self._notify_frames_callbacks(filtered_poses)

        return filtered_poses

    def reset(self) -> None:
        """Reset all pipelines."""
        for pipeline in self._pipelines.values():
            pipeline.reset()

    def reset_at(self, id: int) -> None:
        """Reset pipeline for a specific track."""
        self._pipelines[id].reset()