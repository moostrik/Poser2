"""Tracks pose windows for multiple tracks independently."""

from typing import Callable

from .TrackerBase import TrackerBase
from modules.pose.nodes.windows.WindowNode import WindowNode
from modules.pose.frame import FrameDict, FeatureWindowDict, FeatureWindowDictCallbackMixin

import logging
logger = logging.getLogger(__name__)


class WindowTracker(TrackerBase, FeatureWindowDictCallbackMixin):
    """Tracks multiple poses, maintaining a separate WindowNode for each.

    Each track_id gets its own WindowNode which maintains an independent buffer.
    Windows are automatically reset when their pose is lost.

    Note: This tracker has TWO callback mechanisms:
    - add_frames_callback: Receives input poses (FrameDict) - for chaining with other pose processors
    - add_window_callback: Receives output windows (FeatureWindowDict) - for DataHub/visualization
    """

    def __init__(self, num_tracks: int, window_factory: Callable[[], WindowNode]) -> None:
        """Initialize tracker with a window node per track."""
        TrackerBase.__init__(self)
        FeatureWindowDictCallbackMixin.__init__(self)

        self._window_factory = window_factory
        self._windows: dict[int, WindowNode] = {
            id: window_factory() for id in range(num_tracks)
        }

    def process(self, poses: FrameDict) -> FeatureWindowDict:
        """Process poses through window nodes.

        Returns:
            dict mapping track_id to FeatureWindow (may exclude tracks if emit_partial=False and window not full)
        """

        # Reset windows for poses that are no longer present
        for id in self._windows:
            if id not in poses:
                self.reset_at(id)

        results: FeatureWindowDict = {}

        for id, pose in poses.items():
            try:
                window = self._windows[id].process(pose)
                if window is not None:  # Only add if window was emitted
                    results[id] = window
            except Exception as e:
                logger.error(f"WindowTracker: Error processing pose {id}: {e}")
        # Notify pose callbacks (for downstream pose processing)
        self._notify_frames_callbacks(poses)

        # Notify window callbacks (for DataHub/visualization)
        self._notify_window_callbacks(results)

        return results

    def reset(self) -> None:
        """Reset all window buffers."""
        for window in self._windows.values():
            window.reset()

    def reset_at(self, id: int) -> None:
        """Reset window buffer for a specific track."""
        self._windows[id].reset()
